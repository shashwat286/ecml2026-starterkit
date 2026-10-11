# https://docs.docker.com/reference/build-checks/invalid-default-arg-in-from/
ARG TAG=v4.2.6
FROM ghcr.io/flatland-association/flatland-baselines:${TAG}

# Project code needed by the observation builder and the policy
COPY NewAgentInitObs.py shortestpath.py waypoints.py expert_mask.py config.py ./
COPY agents/ agents/
COPY submission/ submission/
COPY reinforcement_learning/ reinforcement_learning/

ENV POLICY=submission.my_policy.MyPolicy
ENV OBS_BUILDER=submission.my_observation_builder.MyObservationBuilder

# install requirements in env activated in entrypoint
RUN bash entrypoint_generic.sh python -m pip install -r submission/requirements.txt -r reinforcement_learning/requirements.txt