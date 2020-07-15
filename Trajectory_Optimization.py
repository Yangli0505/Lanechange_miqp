
"""
Created on Sat Jun 27 16:36:37 2020

@author: lyxc5
"""

import numpy as np
import scipy.sparse as spa
import math
import miosqp
import rospy

from .Cost_function import CostFunction


class OptimizationProcess(object):
    
    def __init__(self,params,dynamic_map,target_lane_index,var_num):
        
        # Decision variables
        
        self.var_num                                  = var_num
        
        # ego vehicle: x,y,v,theta 
        
        ego_state                                     = self.init_states_egovehicle(dynamic_map)    
        
        
        # Motion Prediction of Surrounding Vehicles
        
        N_pred                                        = params.Np + 1
        
        fv_state_final,tf_state_final,tr_state_final  = self.pred_states_surroundvehicle(dynamic_map,target_lane_index,params,N_pred)
        
        x_tf                                          = tf_state_final[0]
        v_tf                                          = tf_state_final[2]
        
        
        # Initialization of Solution
        
        if target_lane_index == -1:
            
            target_lane = dynamic_map.jmap.reference_path
            
        else:
            
            target_lane = dynamic_map.mmap.lanes[int(target_lane_index)]
   
        y_des       = target_lane.map_lane.central_path_points[0].position.y         # y_des: the centre of the target lane 
        
        lat_dis     = np.abs(ego_state[1]-y_des)
        
        Xp          = self.initialization_solution(params,ego_state,v_tf,lat_dis)
        
        # Optimized Solution
        
        Xs          = self.optimization(dynamic_map,target_lane_index,Xp,params,x_tf)

        '''
        ----------------------------------Outputs------------------------------
        '''
        self.X      = Xs
        
        
    
    def init_states_egovehicle(self, dynamic_map):
        
        '''
        To get initial states of ego vehicle
        x,y,v, theta
        
        '''
        ego_state       = []
        
        ego_vx          = dynamic_map.ego_state.twist.twist.linear.x 
        ego_vy          = dynamic_map.ego_state.twist.twist.linear.y
        
        ego_state.append(dynamic_map.ego_state.pose.pose.position.x)
        ego_state.append(dynamic_map.ego_state.pose.pose.position.y)       
        ego_state.append(math.sqrt(ego_vx*ego_vx + ego_vy*ego_vy))        
        ego_state.append(math.atan2(ego_vy,ego_vx))
        
               
        return ego_state

    def predictCA(init_state,tm):
        
        '''
        States Prediction of surrounding vehicles
        '''
        pre_state = init_state
        
        pre_state.twist.twist.linear.x = init_state.twist.twist.linear.x + init_state.accel.accel.linear.x*tm
        pre_state.twist.twist.linear.y = init_state.twist.twist.linear.y + init_state.accel.accel.linear.y*tm
        pre_state.pose.pose.position.x = init_state.pose.pose.position.x + init_state.twist.twist.linear.x*tm + 0.5*init_state.accel.accel.linear.x*tm*tm
        pre_state.pose.pose.position.y = init_state.pose.pose.position.y + init_state.twist.twist.linear.y*tm + 0.5*init_state.accel.accel.linear.y*tm*tm
        
        return pre_state    
    
    def pred_states_surroundvehicle(self,dynamic_map,target_lane_index,params,ns):
        
        '''
        The states of surrounding vehicle
        '''
        
        DT = params.DT
        

        ego_lane_index_rounded = int(round(dynamic_map.mmap.ego_lane_index))
        rospy.logdebug("ego lane index = %d, lanes number = %d, target lane index = %d",ego_lane_index_rounded,len(dynamic_map.mmap.lanes),target_lane_index)
        
        # The states of surrounding vehicles:
        # the lead car on the ego lane is front_vehicle
        # the lead car on the adjacent lane is  target_front_vehicle
        # the follow car on the adjacent lane is  target_rear_vehicle
        
        # front_vehicle
        
        if len(dynamic_map.mmap.lanes[ego_lane_index_rounded].front_vehicles)>0:
            
            front_vehicle = dynamic_map.mmap.lanes[ego_lane_index_rounded].front_vehicles[0]
            front_vehicle_exist_flag = 1
            
        else:
            front_vehicle_exist_flag = 0
        

        
        # target_front_vehicle
        
        if len(dynamic_map.mmap.lanes[target_lane_index].front_vehicles)>0:
            
            target_front_vehicle = dynamic_map.mmap.lanes[target_lane_index].front_vehicles[0]
            target_front_vehicle_exist_flag = 1
            
        else:
            target_front_vehicle_exist_flag = 0
            
            
        # target_rear_vehicle
        
        if len(dynamic_map.mmap.lanes[target_lane_index].rear_vehicles)>0:
            
            target_rear_vehicle = dynamic_map.mmap.lanes[target_lane_index].rear_vehicles[0]
            target_rear_vehicle_exist_flag = 1
            
        else:
            target_rear_vehicle_exist_flag = 0
                
        
        #  predicted front_vehicle_state
            
        front_vehicle_state_pred            = self.predictCA(front_vehicle.state,ns*DT)
        front_vehicle_state_pred_x          = front_vehicle_state_pred.pose.pose.position.x
        front_vehicle_state_pred_y          = front_vehicle_state_pred.pose.pose.position.y
        front_vehicle_state_pred_vx         = front_vehicle_state_pred.twist.twist.linear.x
        front_vehicle_state_pred_vy         = front_vehicle_state_pred.twist.twist.linear.y
        front_vehicle_state_pred_v          = math.sqrt(front_vehicle_state_pred_vx,front_vehicle_state_pred_vy)
        
        fv_state = []
        fv_state.append(front_vehicle_state_pred_x)
        fv_state.append(front_vehicle_state_pred_y)
        fv_state.append(front_vehicle_state_pred_v)
        
        # predicted target_front_vehicle
        
        target_front_vehicle_state_pred     = self.predictCA(target_front_vehicle.state,ns*DT)
        target_front_vehicle_state_pred_x   = target_front_vehicle_state_pred.pose.pose.position.x
        target_front_vehicle_state_pred_y   = target_front_vehicle_state_pred.pose.pose.position.y
        target_front_vehicle_state_pred_vx  = target_front_vehicle_state_pred.twist.twist.linear.x
        target_front_vehicle_state_pred_vy  = target_front_vehicle_state_pred.twist.twist.linear.y
        target_front_vehicle_state_pred_v   = math.sqrt(target_front_vehicle_state_pred_vx,target_front_vehicle_state_pred_vy)
        
        tf_state = []
        tf_state.append(target_front_vehicle_state_pred_x)
        tf_state.append(target_front_vehicle_state_pred_y)
        tf_state.append(target_front_vehicle_state_pred_v)
        
        # predicted target_rear_vehicle
        
        target_rear_vehicle_state_pred      = self.predictCA(target_rear_vehicle.state,ns*DT)
        target_rear_vehicle_state_pred_x    = target_rear_vehicle_state_pred.pose.pose.position.x
        target_rear_vehicle_state_pred_y    = target_rear_vehicle_state_pred.pose.pose.position.y
        target_rear_vehicle_state_pred_vx   = target_rear_vehicle_state_pred.twist.twist.linear.x
        target_rear_vehicle_state_pred_vy   = target_rear_vehicle_state_pred.twist.twist.linear.y
        target_rear_vehicle_state_pred_v    = math.sqrt(target_rear_vehicle_state_pred_vx,target_rear_vehicle_state_pred_vy)
        
        tr_state = []
        
        tr_state.append(target_rear_vehicle_state_pred_x)
        tr_state.append(target_rear_vehicle_state_pred_y)
        tr_state.append(target_rear_vehicle_state_pred_v)
        
        
        return fv_state,tf_state,tr_state      
    
        
    def initialization_solution(self,params,ego_state,v_tf,lat_dis):
        
        '''
        The lane change trajectory is initialized with a polynominal curve
        
        Input:
            
        # params: parameters   
        # ego_state is the initial state of ego vehicle
        # v_tf is the speed of the target front vehicle
        # lat_dev: lateral deviation of a lane change scenario
         
        Returns:
            X0 (array)
        
        '''

        Np  = params.Np
        
        nx  = params.nx
        nu  = params.nu
        DT  = params.DT
        
        # 5-order polynomial
        te  = Np*DT
        
        T   = np.array([[0,0,0,0,0,1],
             [0,0,0,0,1,0],
             [0,0,0,2,0,0],
             [math.pow(te,5),math.pow(te,4),math.pow(te,3),math.pow(te,2),te,1],
             [5*math.pow(te,4),4*math.pow(te,3),3*math.pow(te,2),2*te,1,0],
             [20*math.pow(te,3),12*math.pow(te,2),6*te,2,0,0]])
        
        # Boundary Condition 
        ## Xq include the initial(position,speed and accel) and final (position,speed and accel)
        
        ve0 = ego_state[2]
        xd  = (ve0 + v_tf)*te*0.5        
        
        Xq  = np.array([0,ve0,0,xd,v_tf,0])
        Yq  = np.array([0,0,0,lat_dis,0,0])
        
        # compute the coefficients of the 5-order polynomial
        Aq  = np.linalg.solve(T,np.transpose(Xq))
        Bq  = np.linalg.solve(T,np.transpose(Yq))
        
        # Initialization of the Solution

        Xinit = np.zeros((self.var_num))
        
        # lane change curve
        
        for i in range(Np+1):
            
            ti      = DT*i
            
            Ti      = np.array([math.pow(ti,5),math.pow(ti,4),math.pow(ti,3),math.pow(ti,2),ti,1])
            x_i     = np.dot(Aq,Ti)
            y_i     = np.dot(Bq,Ti)
            
            Tidot   = np.array([5*math.pow(ti,4),4*math.pow(ti,3),3*math.pow(ti,2),2*ti,1,0])
            vx_i    = np.dot(Aq,Tidot)
            vy_i    = np.dot(Bq,Tidot)  
            
            v_i     = np.sqrt(vx_i**2 + vy_i**2)
            theta_i = math.atan2(vy_i,vx_i)
            
            Tidot2   = np.array([20*math.pow(ti,3),12*math.pow(ti,2),6*math.pow(ti,1),2,0,0])
            ax_i    = np.dot(Aq,Tidot2)
            ay_i    = np.dot(Bq,Tidot2)  
            
            a_i     = np.sqrt(ax_i**2 + ay_i**2)

            # State:x,y,v,theta
            Xinit[i*nx+0] = x_i + ego_state[0]
            Xinit[i*nx+1] = y_i + ego_state[1]
            Xinit[i*nx+2] = v_i
            Xinit[i*nx+3] = theta_i  
            
            # Acceleration
            if i<Np:
                
                Xinit[(Np+1)*nx+i*nu+0] = a_i
       
        return Xinit
            
    
    def model_linearization(self,X,params):
    

        '''
        Model Linearization 
        Then, the linear model [A,b] can describe the state transition at different times
        Thus, the linear model can  represent the equality constraint: Ax = b
        
        Input : Xpre(the solution at previous step), Parameters
        
        Output: A, b 
        
        '''

        
        Np       = params.Np
        nx       = params.nx
        nu       = params.nu
        
        DT       = params.DT
        L        = params.L   
        
        # system model
        
        ceq_num = nx*Np + Np-1
        
        A       = np.zeros((ceq_num,self.var_num))
        b       = np.zeros((ceq_num,1))
        
       
        for i in range(1,Np+1):
            
            
            ego_v          = X[(i-1)*nx+2]
            ego_theta      = X[(i-1)*nx+3]
            ego_delta      = X[(Np+1)*nx+(i-1)*nu+1]
    
    
            ego_next_v     = X[i*nx+2]
            ego_next_theta = X[i*nx+3] 
            
            if i<Np:
                
                ego_next_delta = X[(Np+1)*nx+i*nu+1]
                
            else:
                ego_next_delta = ego_delta
            
            
            # 状态转移关系    
            
            A1 = [[-1,0,-math.cos(ego_theta)*DT/2,ego_v*math.sin(ego_theta)*DT/2],
                   [0,-1,-math.sin(ego_theta)*DT/2,-ego_v*math.cos(ego_theta)*DT/2],
                   [0,0,-1,0],
                   [0,0,-0.5*ego_delta*DT/L,-1]]
            
            A2 = [[1,0,-math.cos(ego_next_theta)*DT/2,ego_next_v*math.sin(ego_next_theta)*DT/2],
                   [0,1,-math.sin(ego_next_theta)*DT/2,-ego_next_v*math.cos(ego_next_theta)*DT/2],
                   [0,0,1,0],
                   [0,0,-0.5*ego_next_delta*DT/L,1]]  
            
            A3 = [[0,0],
                  [0,0],
                  [-DT/2,0],
                  [0,-ego_v/L*(DT/2)]]
            
            A4 = [[0,0],
                  [0,0],
                  [-DT/2,0],
                  [0,-ego_next_v/L*(DT/2)]]
            
            C = [DT/2*ego_v*math.sin(ego_theta)*ego_theta + DT/2*ego_next_v*math.sin(ego_next_theta)*ego_next_theta,
                  -DT/2*ego_v*math.cos(ego_theta)*ego_theta - DT/2*ego_next_v*math.cos(ego_next_theta)*ego_next_theta,
                  0,
                 -DT/2*ego_v*ego_delta/L - DT/2*ego_next_v*ego_next_delta/L]
                
                
            A[(i-1)*nx+0:i*nx,(i-1)*nx+0:i*nx]                                               = A1
            A[(i-1)*nx+0:i*nx,i*nx:(i+1)*nx]                                                 = A2
            A[(i-1)*nx+0:i*nx,(Np+1)*nx+(i-1)*nu+0:(Np+1)*nx+(i-1)*nu+2]                     = A3
            A[(i-1)*nx+0:i*nx,(Np+1)*nx+i*nu+0:(Np+1)*nx+i*nu+2]                             = A4
            
            b[(i-1)*nx+0:i*nx,0] = C
            
            # 控制量关系
            if i < Np:
                
                ### a2/dt-a1/dt-jerk = 0
                
                
                A[Np*nx+i-1,(Np+1)*nx + (i-1)*nu + 0]   = -1/DT   
                A[Np*nx+i-1,(Np+1)*nx + i*nu + 0]       =  1/DT
                A[Np*nx+i-1,(Np+1)*nx + Np*nu + i]      = -1
                          
        
        return A, b
    
    
    
    def get_constraint_matrix(self,Ae,be,params,dynamic_map,target_lane_index):
        

        '''
        To solve a MIQP we need: m.setup(P, q, A, l, u, i_idx, i_l, i_u)
        
        This function returns:qp_A, qp_l, qp_u ,i_idx, i_l, i_u 
        
        '''
            

        '''
        ----------------------------Parameters---------------------------------
        
        '''               
        
        Np         = params.Np
        nx         = params.nx
        nu         = params.nu
        
        THW_S      = params.THW_S
        lat_dev    = params.lat_dev
        
        deltamax   = params.deltamax
        deltamin   = params.deltamin
        amax       = params.amax
        amin       = params.amin        
  
        vmax       = params.vmax
        vmin       = params.vmin
        
         
        veh_width  = params.veh_width
        
        
        # Parameters: big-M method
        
        
        M1    = params.M1
        M2    = params.M2
        M3    = params.M3
        M4    = params.M4
        M5    = params.M5
        
        
        '''
        -----------------------------Constraints--------------------------------
        
        '''        
        
        Ne         = Ae.shape[0]
        
        num_ct     = Ne+9+8*Np+Np+1
        
        qp_A       = np.zeros((num_ct,self.var_num))
        qp_l       = np.zeros((num_ct))
        qp_u       = np.zeros((num_ct))
        
        '''
        states transition      状态转移约束
        '''
        
        qp_A[0:Ne,:]   = Ae        
        qp_l[0:Ne]     = be
        qp_u[0:Ne]     = be
            

        '''
        start point constraint 起点约束
        count: 4
        '''
        
        qp_A[Ne:Ne+1,0]    = 1
        qp_A[Ne+1:Ne+2,1]  = 1
        qp_A[Ne+2:Ne+3,2]  = 1
        qp_A[Ne+3:Ne+4,3]  = 1
                
        ego_state          = self.init_states_egovehicle(dynamic_map)  # ego vehicle 自车初始条件
        
        qp_l[Ne:Ne+1]      = ego_state[0]
        qp_l[Ne+1:Ne+2]    = ego_state[1]
        qp_l[Ne+2:Ne+3]    = ego_state[2]
        qp_l[Ne+3:Ne+4]    = ego_state[3]

        qp_u[Ne:Ne+1]      = ego_state[0]
        qp_u[Ne+1:Ne+2]    = ego_state[1]
        qp_u[Ne+2:Ne+3]    = ego_state[2]
        qp_u[Ne+3:Ne+4]    = ego_state[3]    
        
        
        '''
        end point constraint 终端约束
        count: 5
        '''
        
      
        fv_state_final,tf_state_final,tr_state_final  = self.pred_states_surroundvehicle(dynamic_map,target_lane_index,params,Np+1)
        
        lane_width  = abs(ego_state[1] - tf_state_final[1])
        
        
        qp_A[Ne+4:Ne+5,Np*nx+0] = 1         # safe distance with the target front vehicle
        qp_A[Ne+4:Ne+5,Np*nx+2] = 1*THW_S   
        
        qp_A[Ne+5:Ne+6,Np*nx+1] = 1  
        
        qp_A[Ne+6:Ne+7,Np*nx+2] = 1 
        
        qp_A[Ne+7:Ne+8,Np*nx+3] = 1         
        
        qp_A[Ne+8:Ne+9,Np*nx+0] = 1
               
        
        qp_l[Ne+4:Ne+5]  = -np.inf          
        qp_u[Ne+4:Ne+5]  = tf_state_final[0]       # safe distance: target front vehicle
        
        qp_l[Ne+5:Ne+6]  = tf_state_final[1] - lat_dev    # arrive in the target lane
        qp_u[Ne+5:Ne+6]  = tf_state_final[1] + lat_dev  
         
        qp_l[Ne+6:Ne+7]  = vmin                    # speed limit
        qp_u[Ne+6:Ne+7]  = vmax 
        
        qp_l[Ne+7:Ne+8]  = 0                       # heading angle = 0
        qp_u[Ne+7:Ne+8]  = 0 
        
        qp_l[Ne+8:Ne+9]  = tr_state_final[0] + tr_state_final[2]*THW_S       # safe distance: target back vehicle
        qp_u[Ne+8:Ne+9]  = tf_state_final[0]     
        
        
        
        '''
        The process constraint 过程约束
        '''
        
        for i in range(Np):
            
            '''
            States constraints
            '''
            # lateral position constraint
            
            qp_A[Ne+9+i:Ne+9+i+1,(i+1)*nx + 1]                    = 1                            # y
            qp_l[Ne+9+i:Ne+9+i+1]                                 = ego_state[1] - lane_width
            qp_u[Ne+9+i:Ne+9+i+1]                                 = ego_state[1] + lane_width
            
            qp_A[Ne+9+Np+i:Ne+9+Np+i+1,(i+1)*nx + 2]              = 1            # v 
            qp_l[Ne+9+Np+i:Ne+9+Np+i+1]                           = vmin
            qp_u[Ne+9+Np+i:Ne+9+Np+i+1]                           = vmax
            
            qp_A[Ne+9+2*Np+i:Ne+9+2*Np+i+1,(Np+1)*nx + i*nu + 0]  = 1            # acceleration           
            qp_l[Ne+9+2*Np+i:Ne+9+2*Np+i+1]                       = amin
            qp_u[Ne+9+2*Np+i:Ne+9+2*Np+i+1]                       = amax
            
            
            qp_A[Ne+9+3*Np+i:Ne+9+3*Np+i+1,(Np+1)*nx + i*nu + 1]  = 1           # front wheel angle
            qp_l[Ne+9+3*Np+i:Ne+9+3*Np+i+1]                       = deltamin
            qp_u[Ne+9+3*Np+i:Ne+9+3*Np+i+1]                       = deltamax
            
            
            '''
            big-M contraints
            safe THW related to lane
            '''
            
            
            fv_state =[]
            tf_state =[]
            tr_state =[]
            
            # Prediction of surrounding vehicles
            
            fv_state,tf_state,tr_state  = self.pred_states_surroundvehicle(dynamic_map,target_lane_index,params,i+1)
            
            
            '''c1'''
            # from 2 to (Np+1)
            
            qp_A[Ne+9+4*Np+i:Ne+9+4*Np+i+1,(i+1)*nx + 1]                          = -1
            qp_A[Ne+9+4*Np+i:Ne+9+4*Np+i+1,(Np+1)*nx + Np*nu + Np-1 + i]          = -M1
            
            qp_l[Ne+9+4*Np+i:Ne+9+4*Np+i+1]                                       = -np.inf
            qp_u[Ne+9+4*Np+i:Ne+9+4*Np+i+1]                                       = -0.5*lane_width-0.5*veh_width

            '''c2'''
            
            qp_A[Ne+9+5*Np+i:Ne+9+5*Np+i+1,(i+1)*nx + 0]                          = 1 
            qp_A[Ne+9+5*Np+i:Ne+9+5*Np+i+1,(Np+1)*nx + Np*nu + Np-1 + i]          = M2 
            qp_A[Ne+9+5*Np+i:Ne+9+5*Np+i+1,(i+1)*nx + 2]                          = THW_S
            
            qp_l[Ne+9+5*Np+i:Ne+9+5*Np+i+1]                                       = -np.inf
            qp_u[Ne+9+5*Np+i:Ne+9+5*Np+i+1]                                       =  M2 +fv_state[0]

            '''c3'''
            
            qp_A[Ne+9+6*Np+i:Ne+9+6*Np+i+1,(i+1)*nx + 1]                          = 1
            qp_A[Ne+9+6*Np+i:Ne+9+6*Np+i+1,(Np+1)*nx + Np*nu + Np-1 + Np + i]     = -M3
            
            qp_l[Ne+9+6*Np+i:Ne+9+6*Np+i+1]                                       = -np.inf
            qp_u[Ne+9+6*Np+i:Ne+9+6*Np+i+1]                                       = 0.5*lane_width-0.5*veh_width
            
            '''c4'''
            
            qp_A[Ne+9+7*Np+i:Ne+9+7*Np+i+1,(i+1)*nx + 0]                          = -1
            qp_A[Ne+9+7*Np+i:Ne+9+7*Np+i+1,(Np+1)*nx + Np*nu + Np-1 + Np + i]     = M4
            
            qp_l[Ne+9+7*Np+i:Ne+9+7*Np+i+1]                                       = -np.inf
            qp_u[Ne+9+7*Np+i:Ne+9+7*Np+i+1]                                       = -tr_state[0] - THW_S * tr_state[2] + M4
            
            '''c5'''

            qp_A[Ne+9+8*Np+i:Ne+9+8*Np+i+1,(i+1)*nx + 0]                          = 1
            qp_A[Ne+9+8*Np+i:Ne+9+8*Np+i+1,(Np+1)*nx + Np*nu + Np-1 + Np + i]     = M5
            qp_A[Ne+9+8*Np+i:Ne+9+8*Np+i+1,(i+1)*nx + 2]                          = THW_S
            
            qp_l[Ne+9+8*Np+i:Ne+9+8*Np+i+1]                                       = -np.inf
            qp_u[Ne+9+8*Np+i:Ne+9+8*Np+i+1]                                       =  tf_state[0] + M5    # x2c_obstacle: target front vehicle 
             
        
        ''' Integer Variable'''

        
        i_idx = []
        i_l   = []
        i_u   = []
        
        for i in range(2*Np):
            
            i_idx[i] = nx*(Np+1) + nu*Np + Np-1 + i
            i_l[i]   = 0
            i_u[i]   = 1   
        
        
        return qp_A, qp_l, qp_u, i_idx, i_l, i_u
    
    
    
    def optimization(self,dynamic_map,target_lane_index,Xp,params,x_tf):
            
        """
        Solver 
        # x_tf: the final longitudinal position of target front vehicle
        """
        
        '''cost function    '''
        
        if target_lane_index == -1:
            
            target_lane = dynamic_map.jmap.reference_path
        else:
            
            target_lane = dynamic_map.mmap.lanes[int(target_lane_index)]
            
        # y_des is the centre of the target lane, x_tf is final longitudinal position of target front vehicle
        
        y_des = target_lane.map_lane.central_path_points[0].position.y
        
        obj_fun = CostFunction(params,self.var_num,y_des,x_tf)
        
        qp_P    = obj_fun.P
        qp_q    = obj_fun.q
        
        
        ''' Optimization Loop 
        
            iteration
        '''
        
        flag  = 1
        count = 1
        
        max_iter = params.max_iter
        Np       = params.Np
        nx       = params.nx
        
        
        
        while flag: 
            
            print('================== iter={} ===================='.format(count))
            
            if count > max_iter:
                
                print('Maximum iteration number!')
                
                break
                      
            
            # Model Linearization
            ## based on the solution solved at previous step
        
            Ae, be                             = self.model_linearization(Xp,params)
                        
            qp_A, qp_l, qp_u,i_idx, i_l, i_u   = self.get_constraint_matrix(Ae,be,params,dynamic_map,target_lane_index)
            
            
            # P and A are both in the scipy sparse CSC format.
            
            qp_P  = spa.csc_matrix(qp_P)
             
            qp_A  = spa.csc_matrix(qp_A)
            
            
            # Solver settings   
        
            miosqp_settings = {'eps_int_feas': 1e-03,
                               'max_iter_bb': 1000,
                               'tree_explor_rule': 1,
                               'branching_rule': 0,
                               'verbose': False,
                               'print_interval': 1}
            
            
            osqp_settings   = {'eps_abs': 1e-03,
                               'eps_rel': 1e-03,
                               'eps_prim_inf': 1e-04,
                               'verbose': False}
            
            
            
            mysolver   = miosqp.MIOSQP()
            
            mysolver.setup(qp_P,qp_q,qp_A,qp_l,qp_u,i_idx,i_l,i_u, miosqp_settings,osqp_settings)
               
            
            # Set initial solution
            # The initial guess can speedup the branch-and-bound algorithm significantly.
            # To set an initial feasible solution x0 we can run:m.set_x0(x0)
            
            solution_init = np.empty(self.var_num)
            
            mysolver.set_x0(solution_init)
            
            
            # Solve the problem
            
            res_miosqp = mysolver.solve()
            
                   
            # Solution 
            
            X   = res_miosqp.x

       
            ''' Iteration '''
            
            
            if count == 1:
                
                Xp    = X
                count = count +1
                
            if count > 1:
                
                x_differ = np.zeros((Np+1,1))
                y_differ = np.zeros((Np+1,1))
                
                for i in range(Np+1):
                    
                    x_differ[i] = X[i*nx+0] - Xp[i*nx+0]
                    y_differ[i] = X[i*nx+1] - Xp[i*nx+1]
                    
                ''' stop condition'''
                    
                if max([abs(fi) for fi in x_differ]) <= 0.05 and max([abs(fj) for fj in y_differ]) <= 0.05:
                    
                    flag = 0
                    
                else:
                    
                    Xp = X
                    
                    
        return X