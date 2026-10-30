-- Silver: sleep sessions attributed to the WAKE date (morning), the grain
-- recovery models consume. One row per night.
with raw as (
    select * from read_parquet('../bronze/sleep_sessions.parquet')
)
select
    cast(end_datetime as date) as sleep_date,  -- wake date
    cast(start_datetime as timestamp) as sleep_start,
    cast(end_datetime as timestamp) as sleep_end,
    (
        coalesce(sleep_core_duration_sec, 0)
        + coalesce(sleep_deep_duration_sec, 0)
        + coalesce(sleep_rem_duration_sec, 0)
        + coalesce(sleep_asleep_unspecified_duration_sec, 0)
    ) / 3600.0 as sleep_total_h,
    sleep_core_duration_sec / 3600.0 as sleep_core_h,
    sleep_deep_duration_sec / 3600.0 as sleep_deep_h,
    sleep_rem_duration_sec / 3600.0 as sleep_rem_h,
    sleep_awake_duration_sec / 3600.0 as sleep_awake_h,
    sleep_efficiency,
    number_of_awakenings::int as awakenings,
from raw
