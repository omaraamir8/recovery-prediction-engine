-- Silver: cleaned daily biometric grain, one row per calendar date.
-- Casts, null-safe renames. Source: bronze/daily_metrics.parquet
with raw as (
    select * from read_parquet('../bronze/daily_metrics.parquet')
)
select
    cast(date as date) as metric_date,
    heart_rate_variability_ms as hrv_ms,
    resting_hr_average_bpm as rhr_bpm,
    respiratory_rate_average as resp_rate,
    oxygen_saturation_average as spo2_avg,
    apple_sleeping_wrist_temperature_average as wrist_temp_c,
    step_count::bigint as steps,
    active_energy_burned_kcal as active_kcal,
    basal_energy_burned_kcal as basal_kcal,
    hr_average_bpm as hr_day_avg,
    hr_max_bpm as hr_day_max,
    distance_walking_running_meters as distance_m,
    body_mass_average as body_mass_kg,
from raw
