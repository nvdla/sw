#!/usr/bin/env python3

##
 # Copyright (c) 2019 NVIDIA Corporation.  All rights reserved.
 #
 # NVIDIA Corporation and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA Corporation is strictly prohibited.
##

import struct
import json
import sys
from collections import OrderedDict
import logging

namemaps = []

def read_calibtable_txt2json(calib_file):
    logging.info("calibration input txt: %s" % (calib_file,))
    root = OrderedDict()
    index_name = 0
    index_val = 1
    with open(calib_file) as calibtxt:
        for line in calibtxt:

            line = line.strip()
            logging.debug("")
            logging.debug(">> %s" % line)

            # colon delimited
            fields = line.split(':')
            logging.debug(fields)
            if len(fields) != 2:
                logging.debug("...skipping")
                continue

            layername = fields[index_name].strip()
            float_hex = fields[index_val].strip()
            float_val = struct.unpack('>f', float_hex.decode("hex"))[0]
            logging.debug("%s: %s", layername, float_val)

            vald = OrderedDict()
            vald['scale'] = float_val
            vald['min'] = 0
            vald['max'] = 0
            vald['offset'] = 0
            root[layername] = vald
    return root

def dump_json(json_root, json_file):
    logging.info("calibration output json: %s" % (json_file,))
    json_str = json.dumps(json_root, indent=4)
    with open(json_file, 'w') as fp:
        fp.write(json_str);
        logging.debug(json_str)

if __name__ == "__main__":
    logging.basicConfig(
                format='%(levelname)s: %(message)s',
                level=logging.INFO)
                #level=logging.DEBUG)
    if len(sys.argv) != 3:
        print "usage: calib_txt_to_json.py input_calib_txt output_calib_json"
        exit(1);
    calib_file = sys.argv[1]
    json_file = sys.argv[2]
    json_data = read_calibtable_txt2json(calib_file)
    dump_json(json_data, json_file)

