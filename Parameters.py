# -*- coding: utf-8 -*-
"""
MIQP Parameters

"""
import math

class ModelParameters(object):
    
    def __init__(self):
        
        ''' Parameters '''
        
        self.Np       = 30   # lane change duration
        self.DT       = 0.1  # 1/f
        
        self.THW_S    = 1.8  # safe THW
        
        
        self.nx       = 4
        self.nu       = 2
        
        self.w_y      = 0.002
        self.w_a      = 0.01 
        self.w_delta  = 0.8
        self.w_jerk   = 250
        self.w_cf     = 0.0001
        
        
        self.lat_dev  = 0.001
        
        self.deltamax =  5/180*math.pi
        self.deltamin = -5/180*math.pi
        
        self.amax =  5
        self.amin = -5
        
        self.vmax = 90/3.6
        self.vmin = 0/3.6
        
        self.lane_width  = 3.75        
        self.veh_width   = 1.8
        self.L           = 2.5
        
        self.M1  = 1e5
        self.M2  = 1e5
        self.M3  = 1e5
        self.M4  = 1e5
        self.M5  = 1e5
        
        self.max_iter = 15
        
        
        
        
#        parser = argparse.ArgumentParser()
#        
#        parser.add_argument('--Np',type = int, default = 50, help ='prediction horizon [Np*0.1 sec]')
#        parser.add_argument('--DT', type = float, default = 0.1 , help = 'time interval')    
#        
#        parser.add_argument('--nx',type = int, default = 4, help ='the number of states of ego vehicle')
#        parser.add_argument('--nu',type = int, default = 2, help ='the number of control variable of ego vehicle')    
#        
#        parser.add_argument('--w_y',type = float, default = 0.002 , help = 'the weights of cost function: lateral position y ')
#        parser.add_argument('--w_a', type = float, default = 0.01 , help ='the weights of cost function: acceleration')
#        parser.add_argument('--w_delta',type = float, default = 0.8, help = 'the weights of cost function: front wheel angle')
#        parser.add_argument('--w_jerk',type = float ,default = 250 , help ='the weights of cost function: jerk')
#        
#        
#        parser.add_argument('--THW_S', type = float, default = 1.8, help = 'safe THW threshold')
#        
#        parser.add_argument('--lat_dev', type = float, default = 0.001 , help = 'lateral deviation')
#        
#        
#        parser.add_argument('--deltamax', type = float, default = 5/180*math.pi , help = 'front wheel, upper bound')
#        parser.add_argument('--deltamin', type = float, default = -5/180*math.pi , help = 'front wheel, lower bound')
#    
#        parser.add_argument('--amax', type = float, default = 5 , help = 'acc limit, upper bound')
#        parser.add_argument('--amin', type = float, default = -5, help = 'acc limit, lower bound')
#        
#        parser.add_argument('--vmax', type = float, default = 90/3.6 , help = 'speed limit, upper bound, [m/s]')
#        parser.add_argument('--vmin', type = float, default = 0/3.6, help = 'speed limit, lower bound, [m/s]')    
#        
#        
#        
#        parser.add_argument('--lane_width', type = float, default = 3.75 , help = 'lane width')
#        parser.add_argument('--veh_width', type = float, default = 1.8 , help = 'vehicle width')
#        parser.add_argument('--L', type = float, default = 2.5 , help = 'vehicle wheelbase')
#        
#        
#        
#        parser.add_argument('--M1',type = float, default = 1e5, help ='big-M, M1')
#        parser.add_argument('--M2',type = float, default = 1e5, help ='big-M, M2')
#        parser.add_argument('--M3',type = float, default = 1e5, help ='big-M, M3')
#        parser.add_argument('--M4',type = float, default = 1e5, help ='big-M, M4')
#        parser.add_argument('--M5',type = float, default = 1e5, help ='big-M, M5')            
        
               
        
        