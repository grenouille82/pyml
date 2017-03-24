'''
Created on Jul 5, 2011

@author: marcel
'''

import numpy as np

import httplib
import urllib

import optparse
import tempfile
import zlib
import gzip

import csv
import json


from contextlib import nested
from upgeo.util.metric import NaiveMatrixSearch


host = 'earthquake.usgs.gov'
url = '/hazards/apps/vs30/vs30.php'

class Coordinate(object):
    '''
    '''
    __slots__ = ('_lat',
                 '_lon')
    
    def __init__(self, lat, lon):
        if lat < -90 or lat > 90:
            raise ValueError('latitude must be in range [-90,90]')
        if lat < -180 or lon > 180:
            raise ValueError('longitude must be in range [-180,180]')
        self._lon = lon
        self._lat = lat

    def _get_lat(self):
        return self._lat

    lat = property(fget=_get_lat)

    def _get_lon(self):
        return self._lon
    
    lon = property(fget=_get_lon)


def make_http_request(coord, offset=0):
    '''
    Make a http request of a given coordinate (lat, lon) to VS30 Mapping Server.
    The method returns the filename of a compressed csv file containing the 
    information.
    
    @todo: - exception handling
    '''
    
    params = urllib.urlencode({'json': True, 
                               'site_name': 'upgeo',
                               'slope': 'active',
                               'output_types[]': 'xyz', 
                               'top_left_lat': coord.lat+offset, 
                               'top_left_lon': coord.lon-offset, 
                               'bottom_right_lat': coord.lat-offset, 
                               'bottom_right_lon': coord.lon+offset 
                                })
    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
    
    
    conn = httplib.HTTPConnection(host)
    conn.request("POST", url, params, headers)
    response = conn.getresponse()
    data = response.read()
    input = json.loads(data)
    filename = input['xyz']['url']
    return filename
    


def download_tmpfile(url):
    '''
    Download a file via HTTP request and store the content into a temporary file, 
    which corresponding object is returned
    '''
    webfile = urllib.urlopen(url)
    tmp = tempfile.NamedTemporaryFile(mode='w+b', prefix='upgeo.down.', delete=False)
    tmp.write(webfile.read())
    tmp.flush()
    webfile.close()
    return tmp
    
def extract_vs30(csvfile, coord, k=4):
    '''
    Compute the vs30 value from a csvfile for a given coordinate. The vs30 value
    is averaged by the k nearest neighbors of the coordinate, which is determined
    by the euclidean distance. 
    '''
    if k < 1:
        raise ValueError('k must be greater than 0')
    
    data = np.empty((0,3))
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        data = np.vstack((data, np.asarray(row, dtype='float')))
    
    print data
    
    k = np.minimum(len(data), k)
    points = data[:,1::-1]
    values = data[:,2]
    
    searcher = NaiveMatrixSearch(points)
    neighbors,_ = searcher.query_knn(np.asarray([coord.lat, coord.lon]), k)
    
    ret = np.sum(values[neighbors])/k
    return ret
    
def main():
    parser = optparse.OptionParser()
    
    parser.add_option("-i", "--input", dest="infile", type="string",
                      help=("the input file in csv format containing stations with corresponding "
                            "geographic coordinates (columns: station - lat - lon)"))
    parser.add_option("-o", "--output", dest="outfile", type="string",
                      help=("the output file where result is write in csv format "
                            "(columns: station - vs30)"))
    parser.add_option("-k", "--maxneighbor", dest="k", type="int",
                      help=("the maximum number of neighors used for the vs30 "
                            "computation [default %default]"))
    parser.add_option("-d", "--offset", dest="offset", type="float",
                      help=("the offset of the geographic coordinates [default %default]"))
    parser.add_option("-c", "--delimiter", dest="delimiter", type="string",
                      help=("the delimiter using for parsing and writing csv files "
                            "[default %default]"))
    
    parser.set_defaults(outfile='./out.txt', k=4, offset=0.005, delimiter=',')
    
    opts, _ = parser.parse_args()
    with nested(open(opts.infile, 'rb'), open(opts.outfile, 'wb')) as (fin, fout):
        reader = csv.reader(fin, delimiter=opts.delimiter)
        writer = csv.writer(fout, delimiter=opts.delimiter)
        
        i = 0
        for row in reader:
            i = i+1
            if i > 8214:
                station, coord = encode_row(row)
                filename = make_http_request(coord, opts.offset)
                tmpfile = download_tmpfile(filename)
                value = extract_vs30(gzip.open(tmpfile.name), coord, opts.k)
                
                print 'station={0}, lat={1}, lon={2}, vs30={3}'.format(station, coord.lat, coord.lon, value)
                writer.writerow(decode_row(station, value))
                fout.flush()
                                                 
        

def encode_row(row):
    '''
    @todo: - exception handling
    '''
    station = int(row[0])
    coord = Coordinate(float(row[1]), float(row[2]))
    return (station, coord)

def decode_row(station, value):
    '''
    '''
    return [str(station), str(value)]
    

    
    
    
    

if __name__ == '__main__':
    main()
    