class bcolors:
    PURPLE = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARN = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    PINK = '\033[38;5;201m'
    YELLOW = '\033[38;5;226m'
    LIGHTBLUE = '\033[38;5;33m'


def print_header(print_ansi_esc = True):

    header = f"""
{bcolors.PINK}    ____              _______   ___   __
{bcolors.PINK}   / __ \____ _____  / ____/ | / / | / /
{bcolors.YELLOW}  / /_/ / __ `/ __ \/ / __/  |/ /  |/ / 
{bcolors.YELLOW} / ____/ /_/ / / / / /_/ / /|  / /|  /  
{bcolors.LIGHTBLUE}/_/    \__,_/_/ /_/\____/_/ |_/_/ |_/


{bcolors.PURPLE}Genes and pansexuals - version 0.0.1{bcolors.ENDC}
"""
    print(header)