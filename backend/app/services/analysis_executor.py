from app.services.analysis_toolkit import execute_operation, validate_operation


MAX_OPERATIONS = 3


def execute_analysis_plan(df, plan):
    operations = plan.get("operations") if isinstance(plan, dict) else []
    if not isinstance(operations, list):
        return [
            {
                "operation_type": "plan",
                "status": "error",
                "error": "Analysis plan operations must be a list.",
                "warnings": [],
            }
        ]

    assumptions = plan.get("assumptions") if isinstance(plan.get("assumptions"), list) else []
    results = []

    for operation in operations[:MAX_OPERATIONS]:
        valid, error = validate_operation(operation)
        if not valid:
            operation_type = operation.get("type") if isinstance(operation, dict) else "unknown"
            results.append(
                {
                    "operation_type": operation_type or "unknown",
                    "status": "error",
                    "error": error,
                    "warnings": [],
                }
            )
            continue
        results.append(execute_operation(df, operation, assumptions=assumptions))

    if len(operations) > MAX_OPERATIONS:
        results.append(
            {
                "operation_type": "plan",
                "status": "error",
                "error": f"Only {MAX_OPERATIONS} operations are allowed per request.",
                "warnings": ["Extra operations were ignored."],
            }
        )

    return results
