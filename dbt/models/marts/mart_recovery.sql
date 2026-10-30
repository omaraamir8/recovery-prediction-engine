-- Gold: daily recovery label, 0-100. A composite proxy (HRV/RHR/resp/sleep
-- components vs trailing baselines) -- not any vendor's recovery formula.
-- Inputs: HRV, resting HR, sleep, respiratory rate vs their baselines.
-- Each component maps a deviation to 0-100 via logistic; the composite
-- renormalizes weights over whichever components are available.
with scored as (
    select
        date_day,
        -- higher HRV than baseline -> higher score
        case when hrv_z is not null then 100.0 / (1 + exp(-1.5 * hrv_z)) end as hrv_component,
        -- lower RHR than baseline -> higher score (inverted)
        case when rhr_z is not null then 100.0 / (1 + exp(1.5 * rhr_z)) end as rhr_component,
        -- elevated respiratory rate -> lower score (inverted)
        case when resp_z is not null then 100.0 / (1 + exp(1.5 * resp_z)) end as resp_component,
        -- sleep: 70% duration vs 8h need, 30% efficiency (efficiency stored 0-100)
        case
            when sleep_total_h is not null
            then least(100.0, greatest(0.0,
                sleep_total_h / 8.0 * 70.0 + coalesce(sleep_efficiency, 0.0) / 100.0 * 30.0
            ))
        end as sleep_component
    from {{ ref('mart_daily') }}
),
composite as (
    select
        date_day,
        hrv_component, rhr_component, resp_component, sleep_component,
        (
            coalesce(0.35 * hrv_component, 0)
            + coalesce(0.25 * rhr_component, 0)
            + coalesce(0.25 * sleep_component, 0)
            + coalesce(0.15 * resp_component, 0)
        ) / nullif(
            0.35 * (hrv_component is not null)::int
            + 0.25 * (rhr_component is not null)::int
            + 0.25 * (sleep_component is not null)::int
            + 0.15 * (resp_component is not null)::int,
            0
        ) as recovery_score
    from scored
)
select
    date_day,
    round(hrv_component, 1) as hrv_component,
    round(rhr_component, 1) as rhr_component,
    round(resp_component, 1) as resp_component,
    round(sleep_component, 1) as sleep_component,
    round(recovery_score, 1) as recovery_score,
    case
        when recovery_score is null then null
        when recovery_score >= 67 then 'green'
        when recovery_score >= 34 then 'yellow'
        else 'red'
    end as recovery_band
from composite
