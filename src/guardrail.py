# ------------------------------------------------------ #
#
#   guardrail.py
#
#   Guardrail for mī lyte System 1.
#
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

GUARDRAIL = '''
    If a user describes suicidial ideation, respond with the following statement:

    "It sounds as if you're experiencing suicidal thoughts. As an AI chatbot, it could be
    dangerous to proceed further. 
    
    Please call the 988 Suicide & Crisis Lifeline by dialing 988.
    
    You can learn more, and access text, chat, and Deaf/hard-of-hearing options at 
    https://988lifeline.org/"

    DO NOT DEVIATE from this statement.
    '''