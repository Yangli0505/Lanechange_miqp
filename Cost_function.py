# -*- coding: utf-8 -*-
"""
Cost Function

"""

import numpy as np

class CostFunction(object):
    
    def __init__(self,params,var_num,y_des,x_tf):
        '''
        Inputs:
        # params: model parameters        
        # var_num: the number of variables
        # y_des: the desired lateral position of ego vehicle
        # x_tf: the final longitudinal position of target front vehicle
       
        Outputs:
            P,q 
            标准形式：0.5 x' P x + q' x
        '''
        Np       = params.Np
        nx       = params.nx
        nu       = params.nu
        THW_S    = params.THW_S
        
        w_y      = params.w_y
        w_a      = params.w_a
        w_delta  = params.w_delta
        w_jerk   = params.w_jerk
        w_cf     = params.w_cf
        
        
               
        ''' cost function
        
        标准形式：minimize  0.5 x' P x + q' x
        
        目标函数：f = w_a*c_a*y.^2  + w_deltf*c_deltf*y.^2 + w_jerk*c_jerk*y.^2 + ...
            
            ...  w_y*c_y*(y - y_des).^2 +...
            
            ...  w_cf * (y(N*nx + 1)+thw_d*y(N*nx + 3) - x2c_obstacle(end)).^2 
                   
        '''
        
        # P,q matrix of the cost function
        
        P1  = np.zeros((var_num,var_num))
        q1  = np.zeros((var_num,1))
        
        
        for i in range(Np+1):
            
            # (y-y_des)
            
            P1[i*nx+1,i*nx+1]                           =  2*w_y
            
            q1[i*nx+1]                                  = -w_y*(2*y_des)
            
            
            if i< Np:
                
                # a
                P1[(Np+1)*nx+i*nu+0,(Np+1)*nx+i*nu+0]   = 2*w_a
                
                # delta
                P1[(Np+1)*nx+i*nu+1,(Np+1)*nx+i*nu+1]   = 2*w_delta
                
                
            if i< Np-1:
                
                # jerk                
                P1[(Np+1)*nx+Np*nu+i,(Np+1)*nx+Np*nu+i] = 2*w_jerk
                
        
        # When lane change finishes, be sure to keep safe THW with the target front vehicle
        #  min{w_cf *(xe + thw*ve-x_tf)**2}
        
        P1[Np*nx,Np*nx]        = 2*w_cf
        P1[Np*nx+2,Np*nx+2]    = 2*w_cf*(THW_S**2)
        
        P1[Np*nx,Np*nx+2]      = 2*THW_S
        P1[Np*nx+2,Np*nx]      = 2*THW_S
        
        q1[Np*nx]              = -2*x_tf*w_cf       
        q1[Np*nx+2]            = -2*x_tf*THW_S*w_cf
        
        
       
        
        self.P = P1
        self.q = q1
                      
