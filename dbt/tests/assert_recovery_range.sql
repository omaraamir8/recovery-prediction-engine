-- Recovery score must stay in [0, 100].
select date_day, recovery_score
from {{ ref('mart_recovery') }}
where recovery_score is not null
  and (recovery_score < 0 or recovery_score > 100)
