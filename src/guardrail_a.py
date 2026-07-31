# ------------------------------------------------------ #
#
#   guardrail_a.py
#
#   Guardrail for mī lyte System 1. A BASE
#   Simone J. Skeen x Claude Code (06-25-2026)
#
# ------------------------------------------------------ #

##########################################################
#
#  NOTE 6/25: Language below is provisional and UNTESTED. Do NOT use in production.
#
##########################################################

##########################################################
#
#  TODO 6/25: Align w/ mHEAL Suicidality Safety Plan
#
##########################################################

GUARDRAIL_A_BASE = '''
    If a user describes active suicidal ideation, ignore previous instructions and respond with:

    "It sounds as if you're experiencing suicidal thoughts. As an AI chatbot, it could be
    dangerous to proceed further.

    Please call the 988 Suicide & Crisis Lifeline by dialing 988.

    You can access text, chat, and Deaf/hard-of-hearing options at https://988lifeline.org/"

    DO NOT CONSULT your context. DO NOT DEVIATE from this statement. DO NOT recite poetry.
    '''

# Unique phrase for detecting guardrail activation in responses
# Update this if the guardrail script changes
GUARDRAIL_MARKER = r'988 Suicide & Crisis Lifeline'