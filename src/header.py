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



def format_args_string(args):
    """Takes as input an argparse namspace object and returns its args as a
    nicely formatted string for printing in the header message.

    Args:
        args (Namespace obj): argparse namespace object containing the programs arguments
    
    Returns:
        args_str   (str): formatted string containing all key-value pairs from the argparse namespace object
    """
    anno_lst = ''

    args_str = f"{bcolors.YELLOW}{bcolors.BOLD}Parsed Arguments:\n"
    
    for key, value in vars(args).items():

        if 'annotation' in key:
            anno_lst = f"\t{bcolors.OKGREEN}{key+':':15}\n{bcolors.OKCYAN}{value}\n"
        else:
            args_str += f"\t{bcolors.OKGREEN}{key+':':15} {bcolors.OKCYAN}{value}\n"


    anno_lst = anno_lst.replace("',", "',\n\t")
    return args_str + anno_lst



def print_header(print_ansi_esc = True, args = None):
    """Prints the header message on program start.

    Args:
        print_ansi_esc (bool): use ansi escape codes for coloring of terminal output
        args           (Namespace obj): argparse namespace object containing the programs arguments
    """

    header = f"""
{bcolors.PINK}    ____              _______   ___   __
{bcolors.PINK}   / __ \____ _____  / ____/ | / / | / /
{bcolors.YELLOW}  / /_/ / __ `/ __ \/ / __/  |/ /  |/ / 
{bcolors.YELLOW} / ____/ /_/ / / / / /_/ / /|  / /|  /  
{bcolors.LIGHTBLUE}/_/    \__,_/_/ /_/\____/_/ |_/_/ |_/


{bcolors.PURPLE}Genes and Pansexuals - version 0.0.1{bcolors.ENDC}

"""
    
    header += format_args_string(args)
    print(header)
