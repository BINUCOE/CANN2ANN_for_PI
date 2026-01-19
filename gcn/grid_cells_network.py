import numpy as np

# global VT


class GridCellNetwork:
    def __init__(self, **kwargs):   # define some variables of 3d gc
        
        # The x, y, z dimension of 3D Grid Cells Model (3D CAN)
        self.GC_X_DIM = kwargs.pop("GC_X_DIM", 36)
        self.GC_Y_DIM = kwargs.pop("GC_Y_DIM", 36)
        self.GC_Z_DIM = kwargs.pop("GC_Z_DIM", 36)
        #The x , y, z range of 3D Grid Cells Model
        self.GC_X_RANGE = kwargs.pop("GC_X_RANGE", 36)
        self.GC_Y_RANGE = kwargs.pop("GC_Y_RANGE", 36)
        self.GC_Z_RANGE = kwargs.pop("GC_Z_RANGE", 36)
        # The dimension of local excitation weight matrix for x, y, z
        self.GC_EXCIT_X_DIM = kwargs.pop("GC_EXCIT_X_DIM", 7)
        self.GC_EXCIT_Y_DIM = kwargs.pop("GC_EXCIT_Y_DIM", 7)
        self.GC_EXCIT_Z_DIM = kwargs.pop("GC_EXCIT_Z_DIM", 7)
        # The dimension of local excitation weight matrix for x, y, z
        self.GC_INHIB_X_DIM = kwargs.pop("GC_INHIB_X_DIM", 5)
        self.GC_INHIB_Y_DIM = kwargs.pop("GC_INHIB_Y_DIM", 5)
        self.GC_INHIB_Z_DIM = kwargs.pop("GC_INHIB_Z_DIM", 5)
        #The mode to initialize
        self.center_init= kwargs.pop("center_init" , True)
        # The global inhibition value
        self.GC_GLOBAL_INHIB = kwargs.pop("GC_GLOBAL_INHIB", 0.0002)

        # The amount of energy injected when a view template is re-seen
        self.GC_VT_INJECT_ENERGY = kwargs.pop("GC_VT_INJECT_ENERGY", 0.1)

        # Variance of Excitation and Inhibition in XY and THETA respectively
        self.GC_EXCIT_X_VAR = kwargs.pop("GC_EXCIT_X_VAR", 1.5)
        self.GC_EXCIT_Y_VAR = kwargs.pop("GC_EXCIT_Y_VAR", 1.5)
        self.GC_EXCIT_Z_VAR = kwargs.pop("GC_EXCIT_Z_VAR", 1.5)

        self.GC_INHIB_X_VAR = kwargs.pop("GC_INHIB_X_VAR", 2)
        self.GC_INHIB_Y_VAR = kwargs.pop("GC_INHIB_Y_VAR", 2)
        self.GC_INHIB_Z_VAR = kwargs.pop("GC_INHIB_Z_VAR", 2)
        
        # The scale of horizontal translational velocity
        self.GC_HORI_TRANS_V_SCALE = kwargs.pop("GC_HORI_TRANS_V_SCALE", 1)

        # The scale of vertical translational velocity
        self.GC_VERT_TRANS_V_SCALE = kwargs.pop("GC_VERT_TRANS_V_SCALE", 1)

        # packet size for wrap, the left and right activity cells near
        self.GC_PACKET_SIZE = kwargs.pop("GC_PACKET_SIZE", 4)
        #The x,y,z resolution in 3D grid cell network
        self.GC_X_RES = self.GC_X_RANGE / self.GC_X_DIM
        self.GC_Y_RES = self.GC_Y_RANGE / self.GC_Y_DIM
        self.GC_Z_RES = self.GC_Z_RANGE / self.GC_Z_DIM
        # The weight of excitation in 3D grid cell network
        self.GC_EXCIT_WEIGHT = self.create_gc_weights(self.GC_EXCIT_X_DIM, self.GC_EXCIT_Y_DIM, self.GC_EXCIT_Z_DIM,
                                                      self.GC_EXCIT_X_VAR, self.GC_EXCIT_Y_VAR, self.GC_EXCIT_Z_VAR)

        # The weight of inhibition in 3D grid cell network
        self.GC_INHIB_WEIGHT = self.create_gc_weights(self.GC_INHIB_X_DIM, self.GC_INHIB_Y_DIM, self.GC_INHIB_Z_DIM,
                                                      self.GC_INHIB_X_VAR, self.GC_INHIB_Y_VAR, self.GC_INHIB_Z_VAR)

        # convenience constants
        self.GC_EXCIT_X_DIM_HALF = int(np.floor(self.GC_EXCIT_X_DIM / 2))
        self.GC_EXCIT_Y_DIM_HALF = int(np.floor(self.GC_EXCIT_Y_DIM / 2))
        self.GC_EXCIT_Z_DIM_HALF = int(np.floor(self.GC_EXCIT_Z_DIM / 2))

        self.GC_INHIB_X_DIM_HALF = int(np.floor(self.GC_INHIB_X_DIM / 2))
        self.GC_INHIB_Y_DIM_HALF = int(np.floor(self.GC_INHIB_Y_DIM / 2))
        self.GC_INHIB_Z_DIM_HALF = int(np.floor(self.GC_INHIB_Z_DIM / 2))

        # The excit wrap of x,y,z in 3D grid cell network
        self.GC_EXCIT_X_WRAP = np.concatenate((np.arange(self.GC_X_DIM - self.GC_EXCIT_X_DIM_HALF, self.GC_X_DIM),
                                               np.arange(self.GC_X_DIM), np.arange(self.GC_EXCIT_X_DIM_HALF)))
        self.GC_EXCIT_Y_WRAP = np.concatenate((np.arange(self.GC_Y_DIM - self.GC_EXCIT_Y_DIM_HALF, self.GC_Y_DIM),
                                               np.arange(self.GC_Y_DIM), np.arange(self.GC_EXCIT_Y_DIM_HALF)))
        self.GC_EXCIT_Z_WRAP = np.concatenate((np.arange(self.GC_Z_DIM - self.GC_EXCIT_Z_DIM_HALF, self.GC_Z_DIM),
                                               np.arange(self.GC_Z_DIM), np.arange(self.GC_EXCIT_Z_DIM_HALF)))

        # The inhibit wrap of x,y,z in 3D grid cell network
        self.GC_INHIB_X_WRAP = np.concatenate((np.arange(self.GC_X_DIM - self.GC_INHIB_X_DIM_HALF, self.GC_X_DIM),
                                               np.arange(self.GC_X_DIM), np.arange(self.GC_INHIB_X_DIM_HALF)))
        self.GC_INHIB_Y_WRAP = np.concatenate((np.arange(self.GC_Y_DIM - self.GC_INHIB_Y_DIM_HALF, self.GC_Y_DIM),
                                               np.arange(self.GC_Y_DIM), np.arange(self.GC_INHIB_Y_DIM_HALF)))
        self.GC_INHIB_Z_WRAP = np.concatenate((np.arange(self.GC_Z_DIM - self.GC_INHIB_Z_DIM_HALF, self.GC_Z_DIM),
                                               np.arange(self.GC_Z_DIM), np.arange(self.GC_INHIB_Z_DIM_HALF)))

        # The x, y, z cell size of each unit in meter or unit
        self.GC_X_TH_SIZE = 2 * np.pi / self.GC_X_DIM
        self.GC_Y_TH_SIZE = 2 * np.pi / self.GC_Y_DIM
        self.GC_Z_TH_SIZE = 2 * np.pi / self.GC_Y_DIM

        # The lookups for finding the centre of the gccell in GRIDCELLS by get_gc_xyz()
        self.GC_X_SUM_SIN_LOOKUP = np.sin(np.arange(self.GC_X_DIM) * self.GC_X_TH_SIZE)
        self.GC_X_SUM_COS_LOOKUP = np.cos(np.arange(self.GC_X_DIM) * self.GC_X_TH_SIZE)

        self.GC_Y_SUM_SIN_LOOKUP = np.sin(np.arange(self.GC_Y_DIM) * self.GC_Y_TH_SIZE)
        self.GC_Y_SUM_COS_LOOKUP = np.cos(np.arange(self.GC_Y_DIM) * self.GC_Y_TH_SIZE)

        self.GC_Z_SUM_SIN_LOOKUP = np.sin(np.arange(self.GC_Z_DIM) * self.GC_Z_TH_SIZE)
        self.GC_Z_SUM_COS_LOOKUP = np.cos(np.arange(self.GC_Z_DIM) * self.GC_Z_TH_SIZE)

        # The wrap for finding maximum activity packet
        self.GC_MAX_X_WRAP = np.concatenate((np.arange(self.GC_X_DIM - self.GC_PACKET_SIZE, self.GC_X_DIM),
                                             np.arange(self.GC_X_DIM), np.arange(self.GC_PACKET_SIZE)))
        self.GC_MAX_Y_WRAP = np.concatenate((np.arange(self.GC_Y_DIM - self.GC_PACKET_SIZE, self.GC_Y_DIM),
                                             np.arange(self.GC_Y_DIM), np.arange(self.GC_PACKET_SIZE)))
        self.GC_MAX_Z_WRAP = np.concatenate((np.arange(self.GC_Z_DIM - self.GC_PACKET_SIZE, self.GC_Z_DIM),
                                             np.arange(self.GC_Z_DIM), np.arange(self.GC_PACKET_SIZE)))

        # set the initial position in the grid cell network
        gcX, gcY, gcZ = self.get_gc_initial_pos()
        self.GRIDCELLS = np.zeros((self.GC_X_DIM, self.GC_Y_DIM, self.GC_Z_DIM))
        self.GRIDCELLS[gcX, gcY, gcZ] = 1.0
        self.MAX_ACTIVE_XYZ_PATH = [gcX, gcY, gcZ]
        self.vt_history = []

    def get_gc_initial_pos(self):
        """Set the initial position in the grid cell network."""
        if self.center_init:
            gcX = self.GC_X_DIM // 2
            gcY = self.GC_Y_DIM // 2
            gcZ = self.GC_Z_DIM // 2
            return gcX, gcY, gcZ
        else:
            return 0,0,0


    @staticmethod
    def create_gc_weights(xDim, yDim, zDim, xVar, yVar, zVar):
        """Creates a 3D normalised distribution of size dimension^3 with a variance of var."""

        xDimCentre = int(np.floor(xDim / 2))
        yDimCentre = int(np.floor(yDim / 2))
        zDimCentre = int(np.floor(zDim / 2))
        weight = np.zeros((xDim, yDim, zDim))
    
        # Fill the weight array with the Gaussian distribution
        for z in range(zDim):
            for x in range(xDim):
                for y in range(yDim):
                    weight[x, y, z] = (1.0 / (xVar * np.sqrt(2 * np.pi))) * np.exp(
                        -((x - xDimCentre) ** 2) / (2 * xVar ** 2)) * (1.0 / (yVar * np.sqrt(2 * np.pi))) * np.exp(
                        -((y - yDimCentre) ** 2) / (2 * yVar ** 2)) * (1.0 / (zVar * np.sqrt(2 * np.pi))) * np.exp(
                        -((z - zDimCentre) ** 2) / (2 * zVar ** 2))
    
        # Ensure that it is normalised
        total = np.sum(weight)
        weight /= total
    
        return weight
    
    def get_gc_xyz(self):
        """Find the center of the most active cell in the 3D Grid Cells Network."""
        # Find the max activated cell
        x, y, z = np.unravel_index(self.GRIDCELLS.argmax(), self.GRIDCELLS.shape, order='C')
        
        # Take the max activated cell +- AVG_CELL in 3d space
        tempGridcells = np.zeros((self.GC_X_DIM, self.GC_X_DIM, self.GC_Z_DIM))
        tempGridcells[np.ix_(self.GC_MAX_X_WRAP[x: x + self.GC_PACKET_SIZE * 2 + 1],
                      self.GC_MAX_Y_WRAP[y: y + self.GC_PACKET_SIZE * 2 + 1],
                      self.GC_MAX_Z_WRAP[z: z + self.GC_PACKET_SIZE * 2 + 1])] = \
            self.GRIDCELLS[np.ix_(self.GC_MAX_X_WRAP[x: x + self.GC_PACKET_SIZE * 2 + 1],
                           self.GC_MAX_Y_WRAP[y: y + self.GC_PACKET_SIZE * 2 + 1],
                           self.GC_MAX_Z_WRAP[z: z + self.GC_PACKET_SIZE * 2 + 1])]
        
        xSumSin = np.sum(np.dot(self.GC_X_SUM_SIN_LOOKUP, np.sum(np.sum(tempGridcells, axis=1), axis=1)))
        xSumCos = np.sum(np.dot(self.GC_X_SUM_COS_LOOKUP, np.sum(np.sum(tempGridcells, axis=1), axis=1)))
        
        ySumSin = np.sum(np.dot(self.GC_Y_SUM_SIN_LOOKUP, np.sum(np.sum(tempGridcells, axis=0), axis=1)))
        ySumCos = np.sum(np.dot(self.GC_Y_SUM_COS_LOOKUP, np.sum(np.sum(tempGridcells, axis=0), axis=1)))
        
        zSumSin = np.sum(np.dot(self.GC_Z_SUM_SIN_LOOKUP, np.sum(np.sum(tempGridcells, axis=0), axis=0)))
        zSumCos = np.sum(np.dot(self.GC_Z_SUM_COS_LOOKUP, np.sum(np.sum(tempGridcells, axis=0), axis=0)))
        
        gcX = np.mod(np.arctan2(xSumSin, xSumCos) / self.GC_X_TH_SIZE, self.GC_X_DIM)
        gcY = np.mod(np.arctan2(ySumSin, ySumCos) / self.GC_Y_TH_SIZE, self.GC_Y_DIM)
        gcZ = np.mod(np.arctan2(zSumSin, zSumCos) / self.GC_Z_TH_SIZE, self.GC_Z_DIM)
        
        return gcX, gcY, gcZ

    def path_integration(self , delta_X, delta_Y, delta_Z):
        weight_X = np.mod(np.abs(delta_X) , 1)
        weight_Y = np.mod(np.abs(delta_Y) , 1)
        weight_Z = np.mod(np.abs(delta_Z) , 1)
        if delta_X != 0:
            shift1_X = int(np.sign(delta_X) * np.mod(np.floor(np.abs(delta_X)), self.GC_X_DIM))
            shift2_X = int(np.sign(delta_X) * np.mod(np.ceil(np.abs(delta_X)), self.GC_X_DIM))
            self.GRIDCELLS = (1 - weight_X) * np.roll(self.GRIDCELLS, shift1_X, axis=0) + \
                             weight_X * np.roll(self.GRIDCELLS, shift2_X, axis=0)
        if delta_Y != 0:
            shift1_Y = int(np.sign(delta_Y) * np.mod(np.floor(np.abs(delta_Y)), self.GC_Y_DIM))
            shift2_Y = int(np.sign(delta_Y) * np.mod(np.ceil(np.abs(delta_Y)), self.GC_Y_DIM))
            self.GRIDCELLS = (1 - weight_Y) * np.roll(self.GRIDCELLS , shift1_Y, axis = 1)+\
                weight_Y * np.roll(self.GRIDCELLS, shift2_Y, axis = 1)
        if delta_Z != 0:
            shift1_Z = int(np.sign(delta_Z) * np.mod(np.floor(np.abs(delta_Z)), self.GC_Z_DIM))
            shift2_Z = int(np.sign(delta_Z) * np.mod(np.ceil(np.abs(delta_Z)), self.GC_Z_DIM))
            self.GRIDCELLS = (1 - weight_Z) * np.roll(self.GRIDCELLS , shift1_Z, axis = 2)+\
                weight_Z * np.roll(self.GRIDCELLS, shift2_Z, axis = 2)

    def gc_iteration(self, vt_id, transV, curYawThetaInRadian, heightV, VT):
        """
        3D grid cell update steps
        1. Add view template energy
        2. Local excitation
        3. Local inhibition
        4. Global inhibition
        5. Normalisation
        6. Path Integration (vtrans then vheight)
        """
        # Add visual template energy
        actX = int(min(max(np.round(VT[vt_id].gc_x), 1), self.GC_X_DIM - 1))
        actY = int(min(max(np.round(VT[vt_id].gc_y), 1), self.GC_Y_DIM - 1))
        actZ = int(min(max(np.round(VT[vt_id].gc_z), 1), self.GC_Z_DIM - 1))

        energy = self.GC_VT_INJECT_ENERGY * 1.0 / 30.0 * (30.0 - np.exp(1.2 * VT[vt_id].decay))
        if energy > 0:
            self.GRIDCELLS[actX, actY, actZ] += energy
        # Local excitation      GC_local_excitation = GC elements * GC weights
        gridcell_local_excit_new = np.zeros((self.GC_X_DIM, self.GC_Y_DIM, self.GC_Z_DIM))
        for z in range(self.GC_Z_DIM):
            for x in range(self.GC_X_DIM):
                for y in range(self.GC_Y_DIM):
                    if self.GRIDCELLS[x, y, z] != 0:
                        gridcell_local_excit_new[np.ix_(self.GC_EXCIT_X_WRAP[x: x + self.GC_EXCIT_X_DIM],
                                                        self.GC_EXCIT_Y_WRAP[y: y + self.GC_EXCIT_Y_DIM],
                                                        self.GC_EXCIT_Z_WRAP[z: z + self.GC_EXCIT_Z_DIM])] += \
                            self.GRIDCELLS[x, y, z] * self.GC_EXCIT_WEIGHT
        
        self.GRIDCELLS = gridcell_local_excit_new
        
        # Local inhibition      GC_li = GC_le - GC_le elements * GC weights
        gridcell_local_inhib_new = np.zeros((self.GC_X_DIM, self.GC_Y_DIM, self.GC_Z_DIM))
        for z in range(self.GC_Z_DIM):
            for x in range(self.GC_X_DIM):
                for y in range(self.GC_Y_DIM):
                    if self.GRIDCELLS[x, y, z] != 0:
                        gridcell_local_inhib_new[np.ix_(self.GC_INHIB_X_WRAP[x: x + self.GC_INHIB_X_DIM],
                                                        self.GC_INHIB_Y_WRAP[y: y + self.GC_INHIB_Y_DIM],
                                                        self.GC_INHIB_Z_WRAP[z: z + self.GC_INHIB_Z_DIM])] += \
                            self.GRIDCELLS[x, y, z] * self.GC_INHIB_WEIGHT
        
        self.GRIDCELLS -= gridcell_local_inhib_new
        
        # Global inhibition     Gc_gi = GC_li elements - inhibition
        self.GRIDCELLS = np.where(self.GRIDCELLS >= self.GC_GLOBAL_INHIB, self.GRIDCELLS - self.GC_GLOBAL_INHIB, 0)
        
        # Normalisation
        total = np.sum(self.GRIDCELLS)
        self.GRIDCELLS /= total
        
    # Path integration in x-y plane
        delta_X = -transV * np.sin(curYawThetaInRadian) / self.GC_X_RES
        delta_Y = transV * np.cos(curYawThetaInRadian) / self.GC_Y_RES
        delta_Z = heightV / self.GC_Z_RES
        self.path_integration(delta_X, delta_Y, delta_Z)
            
        self.MAX_ACTIVE_XYZ_PATH = list(self.get_gc_xyz())
        return self.MAX_ACTIVE_XYZ_PATH
