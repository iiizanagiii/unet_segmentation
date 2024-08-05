import numpy as np
import random
from random import randint, choice
from PIL import Image
from typing import Tuple, List

class Rigid_Deform:
    np.seterr(divide='ignore', invalid='ignore')

    def __init__(self, distance: int, points: int):
        self.distance = distance
        self.points = points

    def mls_rigid_deformation(self, vy: np.ndarray, vx: np.ndarray, p: np.ndarray, q: np.ndarray, alpha: float = 1.0, eps: float = 1e-8) -> np.ndarray:

        q = np.ascontiguousarray(q.astype(np.int16))
        p = np.ascontiguousarray(p.astype(np.int16))

        # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
        p, q = q, p

        grow = vx.shape[0]  # grid rows
        gcol = vx.shape[1]  # grid cols
        ctrls = p.shape[0]  # control points

        # Compute
        reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
        reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
        
        w = 1.0 / (np.sum((reshaped_p - reshaped_v).astype(np.float32) ** 2, axis=1) + eps) ** alpha    # [ctrls, grow, gcol]
        w /= np.sum(w, axis=0, keepdims=True)                                               # [ctrls, grow, gcol]

        pstar = np.zeros((2, grow, gcol), np.float32)
        for i in range(ctrls):
            pstar += w[i] * reshaped_p[i]                                                   # [2, grow, gcol]

        vpstar = reshaped_v - pstar                                                         # [2, grow, gcol]
        reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)                                  # [2, 1, grow, gcol]
        neg_vpstar_verti = vpstar[[1, 0],...]                                               # [2, grow, gcol]
        neg_vpstar_verti[1,...] = -neg_vpstar_verti[1,...]                                  
        reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)              # [2, 1, grow, gcol]
        mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)    # [2, 2, grow, gcol]
        reshaped_mul_right = mul_right.reshape(2, 2, grow, gcol)                            # [2, 2, grow, gcol]

        # Calculate q
        reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
        qstar = np.zeros((2, grow, gcol), np.float32)
        for i in range(ctrls):
            qstar += w[i] * reshaped_q[i]                                                   # [2, grow, gcol]
        
        temp = np.zeros((grow, gcol, 2), np.float32)
        for i in range(ctrls):
            phat = reshaped_p[i] - pstar                                                    # [2, grow, gcol]
            reshaped_phat = phat.reshape(1, 2, grow, gcol)                                  # [1, 2, grow, gcol]
            reshaped_w = w[i].reshape(1, 1, grow, gcol)                                     # [1, 1, grow, gcol]
            neg_phat_verti = phat[[1, 0]]                                                   # [2, grow, gcol]
            neg_phat_verti[1] = -neg_phat_verti[1]
            reshaped_neg_phat_verti = neg_phat_verti.reshape(1, 2, grow, gcol)              # [1, 2, grow, gcol]
            mul_left = np.concatenate((reshaped_phat, reshaped_neg_phat_verti), axis=0)     # [2, 2, grow, gcol]
            
            A = np.matmul((reshaped_w * mul_left).transpose(2, 3, 0, 1), 
                            reshaped_mul_right.transpose(2, 3, 0, 1))                       # [grow, gcol, 2, 2]

            qhat = reshaped_q[i] - qstar                                                    # [2, grow, gcol]
            reshaped_qhat = qhat.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)            # [grow, gcol, 1, 2]

            # Get final image transfomer -- 3-D array
            temp += np.matmul(reshaped_qhat, A).reshape(grow, gcol, 2)                      # [grow, gcol, 2]

        temp = temp.transpose(2, 0, 1)                                                      # [2, grow, gcol]
        normed_temp = np.linalg.norm(temp, axis=0, keepdims=True)                           # [1, grow, gcol]
        normed_vpstar = np.linalg.norm(vpstar, axis=0, keepdims=True)                       # [1, grow, gcol]
        transformers = temp / normed_temp * normed_vpstar  + qstar                          # [2, grow, gcol]
        nan_mask = normed_temp[0] == 0

        # Replace nan values by interpolated values
        nan_mask_flat = np.flatnonzero(nan_mask)
        nan_mask_anti_flat = np.flatnonzero(~nan_mask)
        transformers[0][nan_mask] = np.interp(nan_mask_flat, nan_mask_anti_flat, transformers[0][~nan_mask])
        transformers[1][nan_mask] = np.interp(nan_mask_flat, nan_mask_anti_flat, transformers[1][~nan_mask])

        # Remove the points outside the border
        transformers[transformers < 0] = 0
        transformers[0][transformers[0] > grow - 1] = 0
        transformers[1][transformers[1] > gcol - 1] = 0
        
        return transformers.astype(np.int16)

    def demo_auto(self, p: np.ndarray, q: np.ndarray, image: np.ndarray) -> np.ndarray:
        height, width,_= image.shape
        gridX = np.arange(width, dtype=np.int16)
        gridY = np.arange(height, dtype=np.int16)
        vy, vx = np.meshgrid(gridX, gridY)

        rigid = self.mls_rigid_deformation(vy, vx, p, q, alpha=1)
        aug3 = np.ones_like(image)
        aug3[vx, vy] = image[tuple(rigid)]
        return aug3


    def RandMove(self, old_pnt: Tuple[int, int], min_shift: int, max_shift: int) -> Tuple[int, int]:
        neg = [-1,1]

        #get the first point from the geometry object
        old_x = old_pnt[0]
        old_y = old_pnt[1]

        #calculate new coordinates
        new_x = old_x + (choice(neg) * randint(min_shift,max_shift))
        new_y = old_y + (choice(neg) * randint(min_shift,max_shift))

  
        return (new_x,new_y)

    def check_p_q(self, coordinates: List[Tuple[int, int]], distance: int, control_points: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        p_coordinates = []
        q_coordinates = []
        while True:
            if (len(q_coordinates) == control_points):
                break
            else:
                x, y = random.choice(coordinates) 
                old_co = (x,y)
                new_co = self.RandMove(old_co,-distance,distance)
                # print(old_co,new_co)
                # Check if the new coordinates are within the given list of coordinates
                if new_co in coordinates:
                    # print(f"New coordinates:",new_co)
                    p_coordinates.append(old_co)
                    q_coordinates.append(new_co)
                else:
                    pass
                    # print("no")
        return p_coordinates,q_coordinates  
    
    def rigid_deformation(self, image: np.ndarray,mask_image: np.ndarray) -> np.ndarray:
    # Get the height and width of the image
        height, width = image.shape[:2]

        # Set the number of random coordinates you want
        n = 30

        # Generate random coordinates
        random_coordinates = np.random.randint(0, min(height, width), size=(n, 2))

        # Ensure the coordinates are within the image boundaries
        random_coordinates[:, 0] = np.clip(random_coordinates[:, 0], 0, height - 1)
        random_coordinates[:, 1] = np.clip(random_coordinates[:, 1], 0, width - 1)


        P_Points,Q_Points = self.check_p_q(random_coordinates,self.distance,self.points)

        #----- into array
        points_in_p = np.array(P_Points)
        points_in_q = np.array(Q_Points)

        #------ points x,y into y,x-----
        for i in range(len(points_in_p)):
            # for p swap
            temp = points_in_p[i][0]
            points_in_p[i][0] = points_in_p[i][1]
            points_in_p[i][1] = temp

            #for q swap
            temp1 = points_in_q[i][0]
            points_in_q[i][0] = points_in_q[i][1]
            points_in_q[i][1] = temp1
        
            # ------------ Function called -------------
        img_deformation = self.demo_auto(points_in_p,points_in_q,image)
        mask_deformation = self.demo_auto(points_in_p,points_in_q,mask_image)
        img = img_deformation.copy()

        return img,mask_deformation