import time
import os
import datetime
from textwrap import dedent


def st_time(show_func_name=True):
    """
        Decorator to calculate the total time of a func

    Args:
        show_func_name (bool): Whether to show the function name or not
    """

    def wrapper(func):
        def st_func(*args, **keyArgs):
            t1 = time.time()
            r = func(*args, **keyArgs)
            t2 = time.time()
            if show_func_name:
                print("Function=%s, Time elapsed = %ds" % (func.__name__, t2 - t1))
            else:
                print("Time elapsed = %ds" % (t2 - t1))
            return r

        return st_func

    return wrapper


def clear_logs_folder():
    """
        Clear the output directories such
        as output/ and logs/

    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(script_dir, '../logs/')
    for the_file in folder:
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def get_model_timestamp():
    """
        Returns a timestamp string formatted for
        file names
    Returns:
        str: Timestamp string
    """
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%Hh%M')

def print_help(arg_name):
    hlp = f"""\
        {arg_name} -i <inputpath> -o <outputpath> -c <csvfile> -sdr
        Options:
        -i --input "path_to input_directory" 
        Specify the input directory (screen/animal/reconstruction)
        name of the reconstruction folder should contain a special string:
        default is 'DXA' but ic can be changed by -r or -f options
        
        -r --rec 
        Use 'Rec' as a special string indicating the reconstruction folder

        -f "special_string"
        Specify the string indicating te reconstruction folder

        -d --dxa
        The input folder is the reconstruction folder regardless of its name

        -s
        Use models for mice with stretched legs

        -c --csv "path_to_csv"
        Specify path to csv file with list of images to process
    """
    print(dedent(hlp))
