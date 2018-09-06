/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	// Set default random engine
	default_random_engine gen;

	// Set standard deviation values into readable variables
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	// Set the number of particles
	num_particles = 10;

	// Resize particles and weights based on number of particles
	weights.resize(num_particles);

	// Create normal (Gaussian) distribution for p_x, p_y, and p_theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	// Create initial particles
	for(int i = 0; i < num_particles; i++){
		// Insert the random initialization into a particle
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);;
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;

		particles.push_back(particle);
	}

	// Set flag about initialization as true
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	// Set default random engine
	default_random_engine gen;

	// Set standard deviation posterior values into readable variables
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	// Create normal (Gaussian) distribution noise
	normal_distribution<double> dist_noise_x(0.0, std_x);
	normal_distribution<double> dist_noise_y(0.0, std_y);
	normal_distribution<double> dist_noise_theta(0.0, std_theta);
	
	// Calculate prediction for each particle
	for(unsigned int i = 0; i < num_particles; i++){
		if(fabs(yaw_rate) < 0.0001){
			particles[i].x += velocity * delta_t * cos(particles[i].theta) + dist_noise_x(gen);
			particles[i].y += velocity * delta_t * sin(particles[i].theta) + dist_noise_y(gen);
			particles[i].theta += dist_noise_theta(gen);
		} else {
			particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + dist_noise_x(gen);
			particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) + dist_noise_y(gen);
			particles[i].theta += yaw_rate * delta_t + dist_noise_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// Initialize required variables for common purpose
	double variance_x = pow(std_landmark[0], 2);
	double variance_y = pow(std_landmark[1], 2);
	double covariance_xy = std_landmark[0] * std_landmark[1];
	double weights_sum = 0;	

	// Update weight of all particles
	for(int i = 0; i < num_particles; i++) {
		// Find all map landmarks that still within range of sensor range
		Particle& particle = particles[i];

		// Calculate the weight prediction for each particle
		long double final_weight = 1;
		
		for(int j = 0; j < observations.size(); j++) {
			LandmarkObs observation = observations[j];
			
			// Transform by using Homogeneous Transformation
			double trans_x = observation.x * cos(particle.theta) - observation.y * sin(particle.theta) + particle.x;
			double trans_y = observation.x * sin(particle.theta) + observation.y * cos(particle.theta) + particle.y;

			Map::single_landmark_s nearest_landmark;
			double minimum_distance = sensor_range;
			double current_distance = 0;
 
			for(int k = 0; k < map_landmarks.landmark_list.size(); k++) {

				Map::single_landmark_s current_landmark = map_landmarks.landmark_list[k];

				// Calculate the distance between current landmark and transformed observations
				current_distance = fabs(trans_x - current_landmark.x_f) + fabs(trans_y - current_landmark.y_f);

				// If the current distance is nearer than the minimum distance, update it as the new minimum distance and nearest landmark
				if (current_distance < minimum_distance) {
					minimum_distance = current_distance;
					nearest_landmark = current_landmark;
				}


			}

			// Calculate the mean of Multivariate-Gaussian probability density
			double x_difference = trans_x - nearest_landmark.x_f;
			double y_difference = trans_y - nearest_landmark.y_f;
			double probability = exp(-0.5 * ((x_difference * x_difference) / variance_x + (y_difference * y_difference) / variance_y));
			double normalizer = 2 * M_PI * covariance_xy;

			final_weight *= probability/normalizer;

		}

		// Update particle weight
		particle.weight = final_weight;
		
		// Update weight vector
		weights[i] = final_weight;
		
		// Add weight for normalization purpose
		weights_sum += final_weight;

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Set default random engine
	default_random_engine gen;

	discrete_distribution<> probability_distribution(weights.begin(), weights.end());

	// Initialize nre particles vector
	vector<Particle> newParticles;

	// Conduct resampling
	for(unsigned int i = 0; i < num_particles; i++){
		newParticles.push_back(particles[probability_distribution(gen)]);
	}

	// Set resampling result
	particles = newParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
