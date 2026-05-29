import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer 
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    

    # TODO: Perform the optimization and print the result
    result =optimizer.optimize()

    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None 
    best_landmark = None
    sum_of_marginals = float('inf')

    for pose_label, pose in pose_options.items():
        for landmark in [1, 2]:
            graph_landmark = gtsam.NonlinearFactorGraph(graph)
            initial_estimate_landmark = gtsam.Values(initial_estimate)
            graph_landmark, initial_estimate_landmark = add_pose(graph_landmark, initial_estimate_landmark, pose)
            result_landmark = optimize(graph_landmark, initial_estimate_landmark)
            graph_landmark = add_landmark_measurement(graph_landmark, result_landmark, pose, landmark)
            result = optimize(graph_landmark, initial_estimate_landmark)
            marginals_landmark = gtsam.Marginals(graph_landmark, result)
            
            cov1 = marginals_landmark.marginalCovariance(L(1))
            cov2 = marginals_landmark.marginalCovariance(L(2))
            total_uncertainty = np.trace(cov1) + np.trace(cov2)
            
            if total_uncertainty < sum_of_marginals:
                best_pose = pose_label
                best_landmark = landmark
                sum_of_marginals = total_uncertainty

    pose_5 = pose_options[best_pose]
    graph, initial_estimate = add_pose(graph, initial_estimate, pose_5)
    result = optimize(graph, initial_estimate)
    graph = add_landmark_measurement(graph, result, pose_5, best_landmark)
    result = optimize(graph, initial_estimate)

    final_marginals = gtsam.Marginals(graph, result)
    sum_of_marginals_final = final_marginals.marginalCovariance(L(1)).sum() + final_marginals.marginalCovariance(L(2)).sum()

    return best_pose, best_landmark, sum_of_marginals_final

def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.
    best_pose = None      # chosen pose option
    best_landmark = None    # chosen landmark (1 or 2)
    lowest_error = float('inf')
    list_of_errors = []
    # TODO: create a list of errors (each index corresponds to a pose) and add the error of each pose to the list
    list_of_errors = []
    # TODO: compute the sum of the errors and return it along with the best pose and landmark
    for pose_label, pose in pose_options.items():
        min_error_pose = float('inf')
        for landmark in [1,2]:
            graph_landmark = gtsam.NonlinearFactorGraph(graph)
            initial_estimate_landmark = gtsam.Values(initial_estimate)

            graph_landmark, initial_estimate_landmark = add_pose(graph_landmark, initial_estimate_landmark, pose)
            result_landmark = optimize(graph_landmark, initial_estimate_landmark)
            graph_landmark = add_landmark_measurement(graph_landmark, result_landmark, pose, landmark)
            result = optimize(graph_landmark, initial_estimate_landmark)

            error_landmark = graph_landmark.error(result)
            if error_landmark < min_error_pose:
                min_error_pose = error_landmark

            if error_landmark < lowest_error:
                lowest_error = error_landmark
                best_pose = pose_label
                best_landmark = landmark
        list_of_errors.append(min_error_pose)

    pose_5 = pose_options[best_pose]
    graph, initial_estimate = add_pose(graph, initial_estimate, pose_5)
    result = optimize(graph, initial_estimate)
    graph = add_landmark_measurement(graph, result, pose_5, best_landmark)
    result = optimize(graph, initial_estimate)

    sum_of_errors = sum(list_of_errors)
    return best_pose, best_landmark, sum_of_errors 