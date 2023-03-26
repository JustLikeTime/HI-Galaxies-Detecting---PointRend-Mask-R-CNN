# Brandon Engelbrecht 2021/05/11

def sat_extract(folder):
    """
    A function in order to extract the IRNSS and QZS satellites from the 
    geo.txt (Geo-Satellites) and create there own file for their TLE info
    
    Requirements:
        folder - path to where the geo.txt file exits
    """
    file = 'geo.txt'
    
    sat_name = [
        ('QZS', 'qzs'), ('IRNSS', 'irnss')
    ]
    
    with open(folder+file) as f:
        sat_file = f.readlines()
        
    for sat in sat_name:
        sat_write = open(folder+sat[1]+'.txt', 'w')
        
        
        for i, line in enumerate(sat_file):
            if sat[0] in line:
                print sat_file[i]
                sat_write.write(sat_file[i])
                sat_write.write(sat_file[i+1])
                sat_write.write(sat_file[i+2])

        sat_write.close()
        
sat_extract(folder = '2019_03_07_tle/')