# Consistency gap report

## spec_gap (2)
- username [req=REQ-003]: requirement REQ-003 mentions field `username` but no endpoint schema declares it
- password [req=REQ-003]: requirement REQ-003 mentions field `password` but no endpoint schema declares it

## spec_req_conflict (1)
- validation_error(400 vs 401) [req=REQ-003]: sources disagree on the status code: {'contract': '400', 'requirement': '401'}

## contract_only (2)
- not_found [req=REQ-002]: only the built-in contract defines this trigger (status 404)
- not_found [req=REQ-004]: only the built-in contract defines this trigger (status 404)

## unsupported_requirement (2)
- 次将触发账号锁定 10 分钟 [req=REQ-003]: requirement REQ-003 asserts a duration rule (次将触发账号锁定 10 分钟) with no spec support
- 次锁定 10 分钟 [req=REQ-003]: requirement REQ-003 asserts a duration rule (次锁定 10 分钟) with no spec support
