"""Background budgets are independent of live response settings."""


def validate_budget(context, output, label):
    if type(context) is not int or not 256 <= context <= 1048576:
        raise ValueError(f'{label} context must be an integer from 256 to 1048576')
    if type(output) is not int or not 1 <= output < context:
        raise ValueError(f'{label} output limit must be a positive integer smaller than its context')


def check_budget(provider, messages, tools, context, output):
    validate_budget(context, output, 'Background task')
    # Providers pack optional context with their tokenizer before inference.
