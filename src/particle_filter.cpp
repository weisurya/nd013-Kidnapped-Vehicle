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
	double unnorm_weight = 1.0;

	// Set the number of particles
	num_particles = 10;

	// Create normal (Gaussian) distribution for p_x, p_y, and p_theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	// Create initial particles
	for(int i = 0; i < num_particles; ++i){
		// Insert the random initialization into a particle
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);;
		particle.theta = dist_theta(gen);
		particle.weight = unnorm_weight;

		particles.push_back(particle);
		weights.push_back(particle.weight);
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
	double zero_float = 0.0;

	// Create normal (Gaussian) distribution noise
	normal_distribution<double> dist_noise_x(zero_float, std_x);
	normal_distribution<double> dist_noise_y(zero_float, std_y);
	normal_distribution<double> dist_noise_theta(zero_float, std_theta);
	
	// Set minimum threshold for yaw_rate
	if(fabs(yaw_rate) < 0.0001){
		yaw_rate = 0.0001;
	}

	// Calculate prediction for each particle
	for(Particle &particle : particles){
		particle.x += (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta)) + dist_noise_x(gen);
		particle.y += (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t)) + dist_noise_y(gen);
		particle.theta += yaw_rate * delta_t + dist_noise_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(auto &observation : observations){
		double nearest_distance = numeric_limits<double>::max();

		for(auto &prediction : predicted){
			double distance = dist(observation.x, observation.y, prediction.x, prediction.y);
			if(distance < nearest_distance){
				observation.id = prediction.id;
				nearest_distance = distance;
			}
		}
	}
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
	double weight_sum = 0.0;
	
	double unnorm_weight = 1.0;
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];

	// Update weight of all particles
	for(auto &particle : particles){
		// Find all map landmarks that still within range of sensor range
		vector<LandmarkObs> ident_landmarks;

		for(auto &landmark : map_landmarks.landmark_list){
			double distance_to_landmark = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);
			if(distance_to_landmark < sensor_range){
				ident_landmarks.push_back(LandmarkObs({landmark.id_i, landmark.x_f, landmark.y_f}));
			}
		}

		// Transform all observations
		vector<LandmarkObs> transformed_observation;
		int index = 0;
		for(auto &observation : observations){
			// Transform by using Homogeneous Transformation
			double trans_x = particle.x + cos(particle.theta) * observation.x - sin(particle.theta) * observation.y;
			double trans_y = particle.y + sin(particle.theta) * observation.x + cos(particle.theta) * observation.y;
			transformed_observation[index++] = LandmarkObs({observation.id, trans_x, trans_y});
		}

		// Identify the correlation between the observation that was identified in sensor range within all transformed observations
		dataAssociation(ident_landmarks, transformed_observation);

		// Calculate the weight prediction for each particle
		double weight_prediction = 1.0;

		for(auto &observation : transformed_observation){

			LandmarkObs predicted_observation;

			// Identify the coordinate of identified observation
			for(auto &landmark : ident_landmarks){
				if(observation.id == landmark.id){
					predicted_observation = observation;
					break;
				}
			}

			// Calculate the mean of Multivariate-Gaussian probability density
			double normalizer = 2 * M_PI * std_x * std_y;
			double probability = exp(-(pow(observation.x - predicted_observation.x, 2) / (2 * std_x * std_x) + pow(observation.y - predicted_observation.y, 2) / (2 * std_x * std_y)));
			double final_weight = probability / normalizer;
			
			weight_prediction *= final_weight;
		}

		particle.weight = weight_prediction;
	}

	// Normalize the particles' weight
	double normalizer = 0.0;
	// Calculate the sum of all weights
	for(auto &particle : particles){
		normalizer += particle.weight;
	}

	// Conduct normalization
	for(auto &particle : particles){
		particle.weight /= normalizer;
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
	for(auto &newParticle : newParticles){
		newParticle = particles[probability_distribution(gen)];
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
