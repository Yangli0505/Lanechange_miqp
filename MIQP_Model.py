# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:54:02 2020

@author: lyxc5

"""


import numpy as np

from .Trajectory_Optimization import OptimizationProcess

from .Parameters import ModelParameters


class MIQPtrajectory(object):
    
                
    def get_trajectory(self,dynamic_map,target_lane_index,desired_speed):
        
        ''' Optimization '''
        
        params  = ModelParameters()

        # The Number of Variables: var_num
        # x_1,y_1,v_1,theta_1,...,x_Np+1,y_Np+1,v_Np+1,theta_Np+1
        # a_1,d_1,...,a_Np,d_Np
        # j_1,...,j_Np-1
        # za_1,...,za_Np
        # zb_1,...,zb_Np
        
        
        var_num = params.nx * (params.Np + 1) + (params.nu * params.Np) + (params.Np - 1) + (2 * params.Np)
        
        # Optimization
        
        result  = OptimizationProcess(params,dynamic_map,target_lane_index,var_num)
        
        # Solution
        
        X_var   = result.X
        
        
        ''' Lon and Lat Position of Ego Vehicle '''
        
        Nstate     = params.nx * (params.Np + 1)
        
        ego_veh_x  = X_var[0:Nstate:params.nx]
        
        ego_veh_y  = X_var[1:Nstate:params.nx]
        
        trajectory = np.vstack((ego_veh_x,ego_veh_y))

        
        return trajectory
    
     