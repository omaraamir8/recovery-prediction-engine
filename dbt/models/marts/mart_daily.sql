-- Gold: one row per calendar date. Model-ready features with trailing
-- 30-day baselines, z-scores, sleep debt, and strain features:
-- linear TRIMP + Bannister-style exponential TRIMP (own HR data; not any vendor's
-- formula), peak session intensity, 7d/28d rolling loads, acute:chronic ratio,
-- training density, and strain x sleep-debt interaction.
-- Sleep is attributed to the wake date; strain/sleep on date D predict
-- recovery on date D+1 (no leakage: baselines use strictly prior rows).
with spine as (
    select * from {{ ref('stg_daily_metrics') }}
),
sleep_agg as (
    select
        sleep_date,
        sum(sleep_total_h) as sleep_total_h,
        avg(sleep_efficiency) as sleep_efficiency,
        sum(awakenings) as awakenings,
        sum(sleep_deep_h) as sleep_deep_h,
        sum(sleep_rem_h) as sleep_rem_h
    from {{ ref('stg_sleep') }}
    group by 1
),
max_hr as (
    select max(hr_day_max) as hr_max_all from {{ ref('stg_daily_metrics') }}
),
-- Per-workout cardiovascular strain, day grain.
-- strain_trimp: linear HR-reserve x duration (legacy).
-- strain_bannister: Bannister-style exponential TRIMP approximation,
--   duration_min * hr_ratio * 0.64 * exp(1.92 * hr_ratio) with
--   hr_ratio = (avg HR - resting HR) / (all-time max HR - resting HR).
--   Built from his own Apple Health HR data -- NOT any vendor's strain formula.
-- strain_peak_intensity: hardest session's peak HR ratio of the day.
workout_strain as (
    select
        w.workout_date,
        count(*) as workout_count,
        sum(w.duration_min) as workout_min,
        sum(
            w.duration_min * case
                when w.hr_avg is not null and d.rhr_bpm is not null
                    then greatest(
                        (w.hr_avg - d.rhr_bpm)
                        / nullif((select hr_max_all from max_hr) - d.rhr_bpm, 0),
                        0
                    )
                -- fallback when HR missing: scale from energy burned
                else coalesce(w.energy_burned_kcal, 0) / 500.0
            end
        ) as strain_trimp,
        sum(
            case
                when w.hr_avg is not null and d.rhr_bpm is not null
                    then w.duration_min
                        * greatest(
                            (w.hr_avg - d.rhr_bpm)
                            / nullif((select hr_max_all from max_hr) - d.rhr_bpm, 0),
                            0
                        )
                        * 0.64
                        * exp(1.92 * greatest(
                            (w.hr_avg - d.rhr_bpm)
                            / nullif((select hr_max_all from max_hr) - d.rhr_bpm, 0),
                            0
                        ))
                else coalesce(w.energy_burned_kcal, 0) / 500.0
            end
        ) as strain_bannister,
        max(
            case
                when w.hr_max is not null and d.rhr_bpm is not null
                    then greatest(
                        (w.hr_max - d.rhr_bpm)
                        / nullif((select hr_max_all from max_hr) - d.rhr_bpm, 0),
                        0
                    )
            end
        ) as strain_peak_intensity
    from {{ ref('stg_workouts') }} w
    left join {{ ref('stg_daily_metrics') }} d
        on d.metric_date = w.workout_date
    group by 1
),
joined as (
    select
        s.metric_date as date_day,
        s.hrv_ms, s.rhr_bpm, s.resp_rate, s.spo2_avg, s.wrist_temp_c,
        s.steps, s.active_kcal, s.basal_kcal, s.distance_m,
        sl.sleep_total_h, sl.sleep_efficiency, sl.awakenings,
        sl.sleep_deep_h, sl.sleep_rem_h,
        coalesce(ws.workout_count, 0) as workout_count,
        coalesce(ws.workout_min, 0) as workout_min,
        coalesce(ws.strain_trimp, 0) as strain_trimp,
        coalesce(ws.strain_bannister, 0) as strain_bannister,
        ws.strain_peak_intensity
    from spine s
    left join sleep_agg sl on sl.sleep_date = s.metric_date
    left join workout_strain ws on ws.workout_date = s.metric_date
),
with_baselines as (
    select
        *,
        avg(hrv_ms) over w30 as hrv_base,
        stddev_samp(hrv_ms) over w30 as hrv_sd,
        count(hrv_ms) over w30 as hrv_n,
        avg(rhr_bpm) over w30 as rhr_base,
        stddev_samp(rhr_bpm) over w30 as rhr_sd,
        count(rhr_bpm) over w30 as rhr_n,
        avg(resp_rate) over w30 as resp_base,
        stddev_samp(resp_rate) over w30 as resp_sd,
        count(resp_rate) over w30 as resp_n
    from joined
    window w30 as (order by date_day rows between 30 preceding and 1 preceding)
)
select
    date_day,
    hrv_ms, rhr_bpm, resp_rate, spo2_avg, wrist_temp_c,
    steps, active_kcal, basal_kcal, distance_m,
    sleep_total_h, sleep_efficiency, awakenings, sleep_deep_h, sleep_rem_h,
    workout_count, workout_min, strain_trimp, strain_bannister,
    strain_peak_intensity,

    -- trailing baselines (null until 7+ observations)
    case when hrv_n >= 7 then hrv_base end as hrv_base_30d,
    case when rhr_n >= 7 then rhr_base end as rhr_base_30d,
    case when resp_n >= 7 then resp_base end as resp_base_30d,

    -- deviations: the core recovery signal
    case when hrv_n >= 7 and hrv_sd > 0 then (hrv_ms - hrv_base) / hrv_sd end as hrv_z,
    case when rhr_n >= 7 and rhr_sd > 0 then (rhr_bpm - rhr_base) / rhr_sd end as rhr_z,
    case when resp_n >= 7 and resp_sd > 0 then (resp_rate - resp_base) / resp_sd end as resp_z,

    -- sleep debt vs 8h need; 7-day rolling load
    case when sleep_total_h is not null then 8.0 - sleep_total_h end as sleep_debt_h,
    sum(case when sleep_total_h is not null then 8.0 - sleep_total_h end)
        over (order by date_day rows between 6 preceding and current row) as sleep_debt_7d,

    -- cumulative training load: 7d acute, 28d chronic, and their ratio.
    -- All trailing (include current day D; D's features predict D+1).
    sum(strain_trimp)
        over (order by date_day rows between 6 preceding and current row) as strain_7d,
    sum(strain_bannister)
        over (order by date_day rows between 6 preceding and current row) as strain_bannister_7d,
    sum(strain_trimp)
        over (order by date_day rows between 27 preceding and current row) as strain_28d,
    sum(strain_trimp)
        over (order by date_day rows between 6 preceding and current row)
        / nullif(
            sum(strain_trimp)
                over (order by date_day rows between 27 preceding and current row),
            0
        ) as strain_acwr,
    -- training density: active days in trailing 7d
    sum((workout_count > 0)::int)
        over (order by date_day rows between 6 preceding and current row) as training_days_7d,
    -- load x recovery interaction: hard training while under-slept
    coalesce(strain_trimp, 0)
        * coalesce(case when sleep_total_h is not null then 8.0 - sleep_total_h end, 0)
        as strain_sleep_interaction,
    -- relative daily load: active energy vs basal (rest-of-day effort included)
    active_kcal / nullif(basal_kcal, 0) as active_rel
from with_baselines
