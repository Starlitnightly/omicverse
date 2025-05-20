# This file allows separating the most CPU intensive routines from the
# main code.  This allows them to be optimized with Cython.  If you
# don't have Cython, this will run normally.  However, if you use
# Cython, you'll get speed boosts from 10-100x automatically.
#
# THE ONLY CATCH IS THAT IF YOU MODIFY THIS FILE, YOU MUST ALSO MODIFY
# fa2util.pxd TO REFLECT ANY CHANGES IN FUNCTION DEFINITIONS!
#
# Copyright (C) 2017 Bhargav Chippada <bhargavchippada19@gmail.com>
#
# Available under the GPLv3

from math import sqrt
from libc.math cimport sqrt as c_sqrt, pow as c_pow

# Import Cython's features for faster execution
import cython
from cython.operator cimport dereference as deref, preincrement as inc

# This will substitute for the nLayout object
cdef class Node:
    def __init__(self):
        self.mass = 0.0
        self.old_dx = 0.0
        self.old_dy = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.x = 0.0
        self.y = 0.0


# This is not in the original java code, but it makes it easier to deal with edges
cdef class Edge:
    def __init__(self):
        self.node1 = -1
        self.node2 = -1
        self.weight = 0.0


# Here are some functions from ForceFactory.java
# =============================================

# Repulsion function.  `n1` and `n2` should be nodes.  This will
# adjust the dx and dy values of `n1`  `n2`
cdef void linRepulsion(Node n1, Node n2, double coefficient=0):
    cdef double xDist = n1.x - n2.x
    cdef double yDist = n1.y - n2.y
    cdef double distance2 = xDist * xDist + yDist * yDist  # Distance squared

    if distance2 > 0:
        cdef double factor = coefficient * n1.mass * n2.mass / distance2
        n1.dx += xDist * factor
        n1.dy += yDist * factor
        n2.dx -= xDist * factor
        n2.dy -= yDist * factor


# Repulsion function. 'n' is node and 'r' is region
cdef void linRepulsion_region(Node n, Region r, double coefficient=0):
    cdef double xDist = n.x - r.massCenterX
    cdef double yDist = n.y - r.massCenterY
    cdef double distance2 = xDist * xDist + yDist * yDist

    if distance2 > 0:
        cdef double factor = coefficient * n.mass * r.mass / distance2
        n.dx += xDist * factor
        n.dy += yDist * factor


# Gravity repulsion function.  For some reason, gravity was included
# within the linRepulsion function in the original gephi java code,
# which doesn't make any sense (considering a. gravity is unrelated to
# nodes repelling each other, and b. gravity is actually an
# attraction)
cdef void linGravity(Node n, double g):
    cdef double xDist = n.x
    cdef double yDist = n.y
    cdef double distance = c_sqrt(xDist * xDist + yDist * yDist)

    if distance > 0:
        cdef double factor = n.mass * g / distance
        n.dx -= xDist * factor
        n.dy -= yDist * factor


# Strong gravity force function. `n` should be a node, and `g`
# should be a constant by which to apply the force.
cdef void strongGravity(Node n, double g, double coefficient=0):
    cdef double xDist = n.x
    cdef double yDist = n.y

    if xDist != 0 and yDist != 0:
        cdef double factor = coefficient * n.mass * g
        n.dx -= xDist * factor
        n.dy -= yDist * factor


# Attraction function.  `n1` and `n2` should be nodes.  This will
# adjust the dx and dy values of `n1` and `n2`.  It does
# not return anything.
cpdef void linAttraction(Node n1, Node n2, double e, bint distributedAttraction, double coefficient=0):
    cdef double xDist = n1.x - n2.x
    cdef double yDist = n1.y - n2.y
    cdef double factor
    
    if not distributedAttraction:
        factor = -coefficient * e
    else:
        factor = -coefficient * e / n1.mass
        
    n1.dx += xDist * factor
    n1.dy += yDist * factor
    n2.dx -= xDist * factor
    n2.dy -= yDist * factor


# The following functions iterate through the nodes or edges and apply
# the forces directly to the node objects.  These iterations are here
# instead of the main file because Python is slow with loops.
cpdef void apply_repulsion(list nodes, double coefficient, list all_nodes=None):
    """Apply repulsion forces between nodes
    
    Args:
        nodes: List of nodes to calculate repulsion for
        coefficient: Repulsion coefficient
        all_nodes: If provided, calculate repulsion against these nodes, 
                  otherwise use nodes for both source and target
    """
    cdef int i, j
    cdef Node n1, n2
    
    if all_nodes is None:
        all_nodes = nodes
        
    for n1 in nodes:
        for n2 in all_nodes:
            if n1 is not n2:  # Skip self-repulsion
                linRepulsion(n1, n2, coefficient)


cpdef void apply_gravity(list nodes, double gravity, double scalingRatio, bint useStrongGravity=False):
    cdef Node n
    
    if not useStrongGravity:
        for n in nodes:
            linGravity(n, gravity)
    else:
        for n in nodes:
            strongGravity(n, gravity, scalingRatio)


cpdef void apply_attraction(list nodes, list edges, bint distributedAttraction, double coefficient, double edgeWeightInfluence):
    cdef Edge edge
    
    # Optimization, since usually edgeWeightInfluence is 0 or 1, and pow is slow
    if edgeWeightInfluence == 0:
        for edge in edges:
            linAttraction(nodes[edge.node1], nodes[edge.node2], 1, distributedAttraction, coefficient)
    elif edgeWeightInfluence == 1:
        for edge in edges:
            linAttraction(nodes[edge.node1], nodes[edge.node2], edge.weight, distributedAttraction, coefficient)
    else:
        for edge in edges:
            linAttraction(nodes[edge.node1], nodes[edge.node2], c_pow(edge.weight, edgeWeightInfluence),
                          distributedAttraction, coefficient)


# For Barnes Hut Optimization
cdef class Region:
    def __init__(self, nodes):
        self.mass = 0.0
        self.massCenterX = 0.0
        self.massCenterY = 0.0
        self.size = 0.0
        self.nodes = nodes
        self.subregions = []
        self.updateMassAndGeometry()

    cdef void updateMassAndGeometry(self):
        cdef double massSumX, massSumY, distance
        cdef Node n
        
        if len(self.nodes) > 1:
            self.mass = 0
            massSumX = 0
            massSumY = 0
            for n in self.nodes:
                self.mass += n.mass
                massSumX += n.x * n.mass
                massSumY += n.y * n.mass
            self.massCenterX = massSumX / self.mass
            self.massCenterY = massSumY / self.mass

            self.size = 0.0
            for n in self.nodes:
                distance = c_sqrt((n.x - self.massCenterX) ** 2 + (n.y - self.massCenterY) ** 2)
                self.size = max(self.size, 2 * distance)

    cpdef void buildSubRegions(self):
        cdef list topleftNodes, bottomleftNodes, toprightNodes, bottomrightNodes
        cdef Node n
        cdef Region subregion
        
        if len(self.nodes) > 1:
            topleftNodes = []
            bottomleftNodes = []
            toprightNodes = []
            bottomrightNodes = []
            # Optimization: The distribution of self.nodes into 
            # subregions now requires only one for loop. Removed 
            # topNodes and bottomNodes arrays: memory space saving.
            for n in self.nodes:
                if n.x < self.massCenterX:
                    if n.y < self.massCenterY:
                        bottomleftNodes.append(n)
                    else:
                        topleftNodes.append(n)
                else:
                    if n.y < self.massCenterY:
                        bottomrightNodes.append(n)
                    else:
                        toprightNodes.append(n)      

            if len(topleftNodes) > 0:
                if len(topleftNodes) < len(self.nodes):
                    subregion = Region(topleftNodes)
                    self.subregions.append(subregion)
                else:
                    for n in topleftNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            if len(bottomleftNodes) > 0:
                if len(bottomleftNodes) < len(self.nodes):
                    subregion = Region(bottomleftNodes)
                    self.subregions.append(subregion)
                else:
                    for n in bottomleftNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            if len(toprightNodes) > 0:
                if len(toprightNodes) < len(self.nodes):
                    subregion = Region(toprightNodes)
                    self.subregions.append(subregion)
                else:
                    for n in toprightNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            if len(bottomrightNodes) > 0:
                if len(bottomrightNodes) < len(self.nodes):
                    subregion = Region(bottomrightNodes)
                    self.subregions.append(subregion)
                else:
                    for n in bottomrightNodes:
                        subregion = Region([n])
                        self.subregions.append(subregion)

            for subregion in self.subregions:
                subregion.buildSubRegions()

    cdef void applyForce(self, Node n, double theta, double coefficient=0.0):
        """Apply force from region to a specific node
        
        Args:
            n: Node to apply force to
            theta: Barnes-Hut parameter theta
            coefficient: Repulsion coefficient
        """
        cdef double distance, size_ratio
        
        # If the current node is a leaf (i.e. it has no sub-regions),
        # then there is nothing to do (we don't apply forces from
        # regions onto their own nodes).
        if len(self.nodes) <= 1 or n in self.nodes:
            return
            
        # If the node is a single node, we apply the force directly
        if len(self.nodes) == 1:
            linRepulsion(n, self.nodes[0], coefficient)
            return
            
        # Calculate distance between node and region's mass center
        distance = c_sqrt((n.x - self.massCenterX) ** 2 + (n.y - self.massCenterY) ** 2)
        
        # If distance is zero, we have a problem. Add a small jitter
        if distance < 0.0001:
            distance = 0.0001
            
        # If the region is sufficiently far away or is a leaf, we apply 
        # the force directly using the region's mass center as a proxy
        size_ratio = self.size / distance
        if size_ratio < theta:
            linRepulsion_region(n, self, coefficient)
            return
            
        # Otherwise, we recursively apply the force from the subregions
        for subregion in self.subregions:
            subregion.applyForce(n, theta, coefficient)

    cpdef applyForceOnNodes(self, list nodes, double theta, double coefficient=0.0):
        """Apply forces from this region to a set of nodes
        
        Args:
            nodes: List of nodes to apply forces to
            theta: Barnes-Hut parameter theta
            coefficient: Repulsion coefficient
        """
        cdef Node n
        
        for n in nodes:
            self.applyForce(n, theta, coefficient)


# Adjust speed and apply forces step
cpdef dict adjustSpeedAndApplyForces(list nodes, double speed, double speedEfficiency, double jitterTolerance):
    """Adjust the speed and apply the calculated forces to update node positions
    
    Args:
        nodes: List of nodes
        speed: Current speed
        speedEfficiency: Speed efficiency factor
        jitterTolerance: Jitter tolerance
        
    Returns:
        dict: Dict with updated values for speed and other metrics
    """
    cdef double totalSwinging = 0.0
    cdef double totalEffectiveTraction = 0.0
    cdef Node n
    cdef double swinging, estimatedOptimalJitterTolerance, minJT, maxJT
    cdef double jt, minSpeedEfficiency, targetSpeed, maxRise, factor
    
    # Calculate swinging and effective traction
    for n in nodes:
        swinging = sqrt((n.old_dx - n.dx) * (n.old_dx - n.dx) + (n.old_dy - n.dy) * (n.old_dy - n.dy))
        totalSwinging += n.mass * swinging
        totalEffectiveTraction += n.mass * 0.5 * sqrt((n.old_dx + n.dx) * (n.old_dx + n.dx) + (n.old_dy + n.dy) * (n.old_dy + n.dy))
    
    # Optimize jitter tolerance
    # The 'right' jitter tolerance for this network is totalSwinging / totalEffectiveTraction
    # But given the current jitterTolerance, compute the adaption
    estimatedOptimalJitterTolerance = 0.05 * totalSwinging / totalEffectiveTraction
    
    # Limit to reasonable bounds
    minJT = 0.05
    maxJT = 2.0
    jt = jitterTolerance * max(minJT, min(maxJT, estimatedOptimalJitterTolerance))
    
    minSpeedEfficiency = 0.05
    
    # Protection against erratic behavior
    if totalSwinging / totalEffectiveTraction > 2.0:
        # If swinging is big, it's erratic
        if speedEfficiency > minSpeedEfficiency:
            speedEfficiency *= 0.5
        jt = max(jt, jitterTolerance)
    
    targetSpeed = jt * speedEfficiency * totalEffectiveTraction / totalSwinging
    
    # Speed control
    maxRise = 0.5
    speed = speed + min(targetSpeed - speed, maxRise * speed)
    
    # Apply forces
    for n in nodes:
        swinging = sqrt((n.old_dx - n.dx) * (n.old_dx - n.dx) + (n.old_dy - n.dy) * (n.old_dy - n.dy))
        factor = speed / (1.0 + sqrt(speed * swinging))
        
        n.x = n.x + (n.dx * factor)
        n.y = n.y + (n.dy * factor)
    
    return {
        'speed': speed, 
        'speedEfficiency': speedEfficiency,
        'totalSwinging': totalSwinging,
        'totalEffectiveTraction': totalEffectiveTraction,
        'estimatedOptimalJitterTolerance': estimatedOptimalJitterTolerance
    } 