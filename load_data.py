import pathlib
import logging
import os
import sys
import common.initial as initial

def process_image(img_root):
    
    img_data_root = pathlib.Path(img_root)
    logging.debug(img_data_root)

def main():
    workdir = os.path.dirname(sys.argv[0])
    if workdir == "." or workdir == "":
        workdir = os.getcwd()
    os.chdir(workdir)

    # debug option
    from optparse import OptionParser
    usage = """ %prog [options] attachment """
    parser = OptionParser(usage=usage)
    parser.add_option("-X", action="store_true", dest="debug", help="output debug message")
    
    (options, input_args) = parser.parse_args()

    # 设置日志输出格式
    initial.logFormat(options.debug)

    process_image(input_args[1])
