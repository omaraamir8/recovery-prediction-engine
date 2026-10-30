-- Silver: workouts attributed to calendar date of start.
with raw as (
    select * from read_parquet('../bronze/workout_sessions.parquet')
)
select
    cast(start_datetime as date) as workout_date,
    workout_type,
    active_duration_sec / 60.0 as duration_min,
    hr_average_bpm as hr_avg,
    hr_max_bpm as hr_max,
    energy_burned_kcal,
    distance_meters,
    average_speed_mps,
from raw
