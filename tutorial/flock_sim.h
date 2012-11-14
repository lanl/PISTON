/*
Copyright (c) 2012, Los Alamos National Security, LLC
All rights reserved.
Copyright 2012. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL),
which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.

NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.

If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.

Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
·         Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
·         Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other
          materials provided with the distribution.
·         Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used
          to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Christopher Sewell, csewell@lanl.gov
This simulation is based on the method by Matt Sottile described here: http://syntacticsalt.com/2011/03/10/functional-flocks/
*/

#ifndef FLOCK_SIM_H
#define FLOCK_SIM_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/partition.h>
#include <thrust/merge.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random/uniform_real_distribution.h>

#include <math.h>
#include <sys/time.h>
#include <float.h>

#include <iostream>


//===========================================================================
/*!    
    \class      flock_sim

    \brief      
    Performs a flock simulation
*/
//===========================================================================
class flock_sim
{
  public:

    //==========================================================================
    /*! 
        struct cohesion

        Compute cohesion term
    */
    //==========================================================================
    struct cohesion : public thrust::unary_function<float3, float3>
    {
        int n;
        float thresholdSq;
        float3* positions;

        __host__ __device__
        cohesion(int n, float thresholdSq, float3* positions) : n(n), thresholdSq(thresholdSq), positions(positions) { };

        __host__ __device__
        float3 operator()(float3 a_position) const
        {
          // Compute centroid of all neighbors by searching through all other boids
          float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
          int neighbors = 0;
          for (unsigned int i=0; i<n; i++) 
          {
            float3 diff = a_position-positions[i];
            if (dot(diff, diff) < thresholdSq)
            {
              centroid = centroid + positions[i];
              neighbors++;;
            }
          }
          if (neighbors == 0) return make_float3(0.0f, 0.0f, 0.0f);
          centroid.x /= neighbors;  centroid.y /= neighbors;  centroid.z /= neighbors;

          // Add a term to the velocity pointed towards the centroid of the neighbors
          return (centroid - a_position);               
        }
    };


    //==========================================================================
    /*! 
        struct separation

        Compute separation term
    */
    //==========================================================================
    struct separation : public thrust::unary_function<float3, float3>
    {
        int n;
        float thresholdSq;
        float3* positions;

        __host__ __device__
        separation(int n, float thresholdSq, float3* positions) : n(n), thresholdSq(thresholdSq), positions(positions) { };

        __host__ __device__
        float3 operator()(float3 a_position) const
        {
          // Add a term to the velocity pointed away from each neighbor that is too close by searching through all other boids
          float3 repel = make_float3(0.0f, 0.0f, 0.0f);
          int neighbors = 0;
          for (unsigned int i=0; i<n; i++) 
          {
            float3 diff = a_position-positions[i];
            if (dot(diff, diff) < thresholdSq)
            {
              repel = repel + diff;
              neighbors++;
            }
          }
          if (dot(repel, repel) < 0.0000001f) return make_float3(0.0f, 0.0f, 0.0f);
          return normalize(repel);                
        }
    };


    //==========================================================================
    /*! 
        struct alignment

        Compute alignment term
    */
    //==========================================================================
    struct alignment : public thrust::unary_function<thrust::tuple<float3, float3>, float3>
    {
        int n;
        float thresholdSq;
        float3* positions;
        float3* velocities;

        __host__ __device__
        alignment(int n, float thresholdSq, float3* positions, float3* velocities) : n(n), thresholdSq(thresholdSq), positions(positions), velocities(velocities) { };

        __host__ __device__
        float3 operator()(thrust::tuple<float3, float3> a_posAndVel) const
        {
          // Extract the position and the velocity from the tuple
          float3 a_position = thrust::get<0>(a_posAndVel);
          float3 a_velocity = thrust::get<1>(a_posAndVel);

          // Compute the average velocity for all neighbors by searching through all other boids
          float3 avgVelocity = make_float3(0.0f, 0.0f, 0.0f);
          int neighbors = 0;
          for (unsigned int i=0; i<n; i++) 
          {
            float3 diff = a_position-positions[i];
            if (dot(diff, diff) < thresholdSq)
            {
              avgVelocity = avgVelocity + velocities[i];
              neighbors++;
            }
          }
          if (neighbors == 0) return make_float3(0.0f, 0.0f, 0.0f);
          avgVelocity.x /= neighbors;  avgVelocity.y /= neighbors;  avgVelocity.z /= neighbors;

          // Add a term to the velocity to make it closer to the average velocity of the neighbors
          return (avgVelocity - a_velocity);   
        }
    };


    //==========================================================================
    /*! 
        struct updateVelocity

        Compute new velocities
    */
    //==========================================================================
    struct updateVelocity : public thrust::unary_function<int, float3>
    {
        float cohesionWeight, separationWeight, alignmentWeight, velocityAdjustmentScale;
        float3 *cohesion, *separation, *alignment, *velocities;
        float* speeds;

        __host__ __device__
        updateVelocity(float cohesionWeight, float separationWeight, float alignmentWeight, float velocityAdjustmentScale,
                       float3* cohesion, float3* separation, float3* alignment, float3* velocities, float* speeds) : 
                       cohesionWeight(cohesionWeight), separationWeight(separationWeight), alignmentWeight(alignmentWeight), velocityAdjustmentScale(velocityAdjustmentScale),
                       cohesion(cohesion), separation(separation), alignment(alignment), velocities(velocities), speeds(speeds) { };

        __host__ __device__
        float3 operator()(int i) const
        {
          // Adjust the velocity based on the cohesion, separation, and alignment terms and their weights
          float3 newVelocity = (velocities[i] + velocityAdjustmentScale*(cohesionWeight*cohesion[i] + separationWeight*separation[i] + alignmentWeight*alignment[i]));
          speeds[i] = dot(newVelocity, newVelocity);
          return newVelocity;                 
        }
    };


    //==========================================================================
    /*! 
        struct updatePosition

        Compute new positions
    */
    //==========================================================================
    struct updatePosition : public thrust::unary_function<thrust::tuple<float3, float3>, float3>
    {
        float velocityScale, minSpeed, maxSpeed;

        __host__ __device__
        updatePosition(float velocityScale, float minSpeed, float maxSpeed) : velocityScale(velocityScale), minSpeed(minSpeed), maxSpeed(maxSpeed) {};

        __host__ __device__
        float3 operator()(thrust::tuple<float3, float3> a_posAndVel) const
        {
          // Extract the position and the velocity from the tuple, and clamp the velocity between mimimum and maximum values
          float3 a_position = thrust::get<0>(a_posAndVel);
          float3 a_velocity = thrust::get<1>(a_posAndVel);
          if (dot(a_velocity, a_velocity) > maxSpeed*maxSpeed) a_velocity = maxSpeed*normalize(a_velocity);
          if (dot(a_velocity, a_velocity) < minSpeed*minSpeed) a_velocity = minSpeed*normalize(a_velocity);

          // Update the position based on the velocity computed by this timestep
          return (a_position + velocityScale*a_velocity);               
        }
    };


    //==========================================================================
    /*! 
        struct bounce

        Bounce off the boundaries
    */
    //==========================================================================
    struct bounce : public thrust::unary_function<int, void>
    {
        float3 clampMin, clampMax;
        float3* positions;
        float3* velocities;

        __host__ __device__
        bounce(float3 clampMin, float3 clampMax, float3* positions, float3* velocities) : 
              clampMin(clampMin), clampMax(clampMax), positions(positions), velocities(velocities) { };

        __host__ __device__
        void operator()(int i) const
        {
          // If the boid has moved outside the simulation boundaries, clamp it inside and reverse its velocity
          float3 result = positions[i];
          bool bounce = false;
          if (result.x < clampMin.x) { bounce = true; result.x = clampMin.x; }
          if (result.x > clampMax.x) { bounce = true; result.x = clampMax.x; }
          if (result.y < clampMin.y) { bounce = true; result.y = clampMin.y; }
          if (result.y > clampMax.y) { bounce = true; result.y = clampMax.y; }
          if (result.z < clampMin.z) { bounce = true; result.z = clampMin.z; }
          if (result.z > clampMax.z) { bounce = true; result.z = clampMax.z; } 
          positions[i] = result;    
          if (bounce) velocities[i] = -1.0f*velocities[i];           
        }
    };


    //==========================================================================
    /*! 
        Member variable declarations
    */
    //==========================================================================

    //! Boid positions
    thrust::device_vector<float3> m_positions;
    //! Boid velocities
    thrust::device_vector<float3> m_velocities;
    //! Boid speeds
    thrust::device_vector<float>  m_speeds;
    //! Cohesion term
    thrust::device_vector<float3> m_cohesion;
    //! Separation term
    thrust::device_vector<float3> m_separation;
    //! Alignment term
    thrust::device_vector<float3> m_alignment;
    //! Weighting factors for the terms
    float m_cohesionWeight, m_separationWeight, m_alignmentWeight;
    //! Scaling factors
    float m_velocityAdjustmentScale, m_velocityScale;
    //! Boundaries
    float3 m_boundaryMin, m_boundaryMax;
    //! Input size
    int m_n;
    //! Maximum neighborhood distance for cohesion
    float m_cohesionThresholdSq;
    //! Maximum distance for separation
    float m_separationThresholdSq;
    //! Maximum neighborhood distance for alignment
    float m_alignmentThresholdSq;
    //! Scalar range
    float m_scalarMin, m_scalarMax;
    //! Minimum and maximum speeds
    float m_minSpeed, m_maxSpeed;


    //==========================================================================
    /*! 
        Constructor for flock_sim class

        \fn	flock_sim::flock_sim
    */
    //==========================================================================
    flock_sim(thrust::device_vector<float3>& a_inputPositions, thrust::device_vector<float3>& a_inputVelocities, float3 a_boundaryMin, float3 a_boundaryMax,
              float a_cohesionWeight=1.0f, float a_separationWeight=1.0f, float a_alignmentWeight=1.0f, float a_velocityAdjustmentScale=0.01f, float a_velocityScale=1.025f,
              float a_cohesionThreshold=30.0f, float a_separationThreshold=5.0f, float a_alignmentThreshold=30.0f, float a_scalarMin=0.0f, float a_scalarMax=1.0f,
              float a_minSpeed=0.1f, float a_maxSpeed = 5.0f) 
    { 
        // Initialize variables based on input parameters
        m_boundaryMin = a_boundaryMin;  m_boundaryMax = a_boundaryMax;
        m_cohesionThresholdSq = a_cohesionThreshold*a_cohesionThreshold;
        m_separationThresholdSq = a_separationThreshold*a_separationThreshold;
        m_alignmentThresholdSq = a_alignmentThreshold*a_alignmentThreshold;
        m_cohesionWeight = a_cohesionWeight;  m_separationWeight = a_separationWeight;  m_alignmentWeight = a_alignmentWeight;
        m_velocityAdjustmentScale = a_velocityAdjustmentScale;  m_velocityScale = a_velocityScale; 
        m_minSpeed = a_minSpeed;  m_maxSpeed = a_maxSpeed;
        m_scalarMin = a_scalarMin;  m_scalarMax = a_scalarMax;
        m_n = a_inputPositions.size();

        // Allocate memory for device vectors 
        m_positions.resize(m_n);  m_velocities.resize(m_n);  m_speeds.resize(m_n);
        m_cohesion.resize(m_n);  m_separation.resize(m_n);  m_alignment.resize(m_n);   

        // Set initial positions and velocities for the boids   
        thrust::copy(a_inputPositions.begin(), a_inputPositions.end(), m_positions.begin()); 
        thrust::copy(a_inputVelocities.begin(), a_inputVelocities.end(), m_velocities.begin()); 
        thrust::fill(m_speeds.begin(), m_speeds.end(), 0.0f);
    }


    //==========================================================================
    /*! 
        Take a simulation step

        \fn	flock_sim::operator
    */
    //==========================================================================
    void operator()()
    {
        // Compute the cohesion term for the velocity update
        thrust::transform(m_positions.begin(), m_positions.end(), m_cohesion.begin(), 
                          cohesion(m_n, m_cohesionThresholdSq, thrust::raw_pointer_cast(&*m_positions.begin())));

        // Compute the separation term for the velocity update
        thrust::transform(m_positions.begin(), m_positions.end(), m_separation.begin(), 
                          separation(m_n, m_separationThresholdSq, thrust::raw_pointer_cast(&*m_positions.begin())));

        // Compute the alignment term for the velocity update
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(m_positions.begin(), m_velocities.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(m_positions.end(), m_velocities.end())),
                          m_alignment.begin(), 
                          alignment(m_n, m_alignmentThresholdSq, thrust::raw_pointer_cast(&*m_positions.begin()),
                                    thrust::raw_pointer_cast(&*m_velocities.begin())));

        // Update the velocity based on the computed cohesion, separation, and alignment adjustments
        thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+m_n, m_velocities.begin(), 
                          updateVelocity(m_cohesionWeight, m_separationWeight, m_alignmentWeight, m_velocityScale,
                                         thrust::raw_pointer_cast(&*m_cohesion.begin()),
                                         thrust::raw_pointer_cast(&*m_separation.begin()),
                                         thrust::raw_pointer_cast(&*m_alignment.begin()),
                                         thrust::raw_pointer_cast(&*m_velocities.begin()),
                                         thrust::raw_pointer_cast(&*m_speeds.begin())));

        // Update the boid positions based on the new velocities for this time step
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(m_positions.begin(), m_velocities.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(m_positions.end(), m_velocities.end())),
                          m_positions.begin(), updatePosition(m_velocityScale, m_minSpeed, m_maxSpeed));

        // Clamp any boids that have moved outside the simulation boundaries, and reverse their velocities so they bounce back inside                        
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+m_n,
                         bounce(m_boundaryMin, m_boundaryMax, thrust::raw_pointer_cast(&*m_positions.begin()),
                               thrust::raw_pointer_cast(&*m_velocities.begin())));

        // Scale the computed speeds to be between the minimum and maximum scalar values
        float maxSpeed = thrust::reduce(m_speeds.begin(), m_speeds.end(), FLT_MIN, thrust::maximum<float>());
        thrust::transform(m_speeds.begin(), m_speeds.end(), thrust::make_constant_iterator((m_scalarMax-m_scalarMin)/maxSpeed), m_speeds.begin(), thrust::multiplies<float>());
        thrust::transform(m_speeds.begin(), m_speeds.end(), thrust::make_constant_iterator(m_scalarMin), m_speeds.begin(), thrust::plus<float>());
    }

 
    //==========================================================================
    /*! 
        Accessor functions for positions, velocities, and speeds
    */
    //==========================================================================
    thrust::device_vector<float3>::iterator positions_begin()  { return m_positions.begin();  }
    thrust::device_vector<float3>::iterator positions_end()    { return m_positions.end();    } 
    thrust::device_vector<float3>::iterator velocities_begin() { return m_velocities.begin(); }
    thrust::device_vector<float3>::iterator velocities_end()   { return m_velocities.end();   } 
    thrust::device_vector<float>::iterator speeds_begin()      { return m_speeds.begin();     }
    thrust::device_vector<float>::iterator speeds_end()        { return m_speeds.end();       }  
    float get_scalar_min() { return m_scalarMin; }
    float get_scalar_max() { return m_scalarMax; } 
};

#endif



