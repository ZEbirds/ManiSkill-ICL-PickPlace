# Task: Build a Paired In-Context Learning Dataset Generator in ManiSkill 3

## 1. Project Goal

We are using ManiSkill 3 to build a simulation dataset for studying visual/action in-context learning (ICL) in robotic manipulation.

The first task is a simple multi-object Pick-and-Place task:

- A robot arm, preferably the existing Panda robot, operates on a tabletop.
- Multiple visually distinguishable objects are placed on the table.
- There is an open-top basket, box, or other receptacle beside the objects.
- The robot picks up the objects one by one and places them inside the receptacle.
- A Context demonstration shows the order in which the objects are picked and placed.
- A paired Query scene contains the same set of objects, but their initial positions and orientations are changed. The robot should transfer the demonstrated object order to the new scene.

The main research question is whether a model can use a visual/action demonstration to infer an object sequence and execute that sequence in a different scene configuration.

**Language grounding must not be part of this initial experiment.** Do not use natural-language instructions to specify object identities, target objects, or grasp order. The order should be inferred from the temporal sequence of the Context demonstration.

The immediate goal is a working data-generation pipeline, not training a VLA model.

## 2. First Inspect the Existing Repository

Before implementing anything, inspect the actual ManiSkill version, repository structure, existing task implementations, installed dependencies, asset paths, and motion-planning examples.

The following have already been tested in this environment:

1. The official `PickCube-v1` motion-planning example runs successfully and saves a video.
2. `PickSingleYCB-v1` can be instantiated without opening a GUI, using the random-action example.
3. The YCB assets have been downloaded to the default ManiSkill asset directory.
4. Opening a GUI with `--render-mode="human"` fails in the current environment. Prefer offscreen rendering and video/image recording.

Use the existing repository APIs and examples rather than assuming that APIs from a different ManiSkill version are available.

Prioritize reusing existing robot models, controllers, motion-planning utilities, asset loaders, camera interfaces, and demonstration-recording code.

Do not modify core ManiSkill code unless necessary. Prefer a self-contained implementation in a dedicated task/example directory.

## 3. Task Design

### 3.1 Initial scene

Implement a tabletop manipulation scene containing:

- One Panda robot with a parallel-jaw gripper.
- Three visually distinguishable objects.
- One open-top receptacle beside the objects.
- One fixed external RGB camera with a clear view of the robot, all objects, and the receptacle.
- Optional wrist-camera support if it can be added without significantly increasing implementation complexity.

For the first working version, use three simple, reliably graspable objects with visually distinct shapes and/or colors if this makes robust grasping easier.

Prefer these objects initially:
- A cube.
- A cylinder.
- A rectangular cuboid.

The implementation should allow these objects to be replaced with YCB objects or other imported assets later.

Do not attempt to support every arbitrary 3D model in the first version.

### 3.2 Physical task

For each episode:

1. Randomize the initial positions and, where appropriate, orientations of the three objects.
2. Randomize the receptacle position within a valid tabletop region.
3. Generate a target object sequence.
4. Physically grasp the first object in the sequence.
5. Lift and transport it to the receptacle.
6. Release the object inside the receptacle and verify that it remains there.
7. Repeat for the remaining objects in the prescribed order.
8. Mark the episode successful only if every object has been placed inside the receptacle in the required sequence.

Objects must be manipulated through actual robot actions and physics.

Do not teleport objects, directly overwrite their poses to simulate successful manipulation, or mark an object as grasped solely because the robot reached a target pose.

Use the available motion-planning and controller infrastructure where possible. If the existing planner does not support the multi-object task directly, implement a reusable sequential pick-and-place routine using the available IK, motion-planning, and gripper-control interfaces.

Use deterministic, robust scripted policies for expert demonstrations. The demonstrations are intended to provide reliable data, not to optimize task performance.

### 3.3 Object identity and sequence

Each object must have a stable semantic asset key in metadata, such as `cube_red`, `cylinder_blue`, or `cuboid_green`.

These keys are metadata only. Do not render the keys or textual labels into the observation images.

Within a Context–Query pair:
- Use the same set of object types and appearance identities in both scenes.
- Change the positions and optionally orientations of the objects between Context and Query.
- Preferably change the receptacle position as well, while keeping the task feasible.
- Use distinct objects so the model can visually distinguish them.
- Randomly sample a valid object order for each pair; do not use one fixed global order for all episodes.

For example, if the Context picks up objects in the order:

`cylinder_blue -> cube_red -> cuboid_green`

then the paired Query should contain those same three object identities at different positions and demonstrate the same required order.

The Query ground-truth trajectory should follow the order inferred from its paired Context.

Store the order explicitly in metadata, but never provide this metadata to the model as part of its visual observation or action input.

The task should use no changing natural-language instructions. All experimental conditions must be identical with respect to language input; preferably omit language entirely if the interfaces support it.

## 4. Context–Query Pair Generation

Build a generator that creates paired episodes.

Each pair consists of:

### Context episode

A successful expert demonstration containing:
- RGB observation sequence.
- Time-aligned robot action sequence.
- Robot state and gripper state.
- Object state throughout the episode.
- Events identifying each successful grasp and placement.
- The order in which objects were picked and placed.

### Query episode

A separate scene containing:
- The same set of object identities.
- Different initial positions and, optionally, orientations.
- A potentially different receptacle position.
- The same target order as its paired Context.
- Its own successful expert trajectory for offline evaluation.

The Query expert trajectory and Query ground-truth states are evaluation targets only. They must not be included in the model's Query observation input.

Ensure the two scenes are independently initialized. Do not simply replay the Context trajectory with shifted coordinates.

At minimum, provide a configurable number of pairs and independent random seeds for reproducibility.

Recommended initial setting:
- 3 objects per scene.
- 2–5 pairs for a smoke test.
- 20–50 pairs for a small pilot dataset, once the pipeline is stable.

Do not generate a large dataset before verifying that trajectories are physically successful and observations are correctly synchronized.

## 5. Data Collection Requirements — LeRobot v3 Format

**All dataset outputs must use the LeRobot v3 dataset format.** Do not create a separate custom dataset format as the primary output. Custom metadata or auxiliary files are allowed only when needed, and must remain compatible with the LeRobot v3 dataset structure.

Before implementation, inspect the installed LeRobot version and the official LeRobot v3 dataset specification. Use the actual APIs and schema supported by the installed version; do not assume that APIs from older LeRobot versions are compatible.

### 5.1 Dataset structure and serialization

- Export the dataset in LeRobot v3 format, using the official dataset writer/API where available.
- Follow the official v3 storage conventions for episode metadata, tabular data, video streams, features, and episode indexing.
- Include RGB observations and time-aligned robot action/state data.
- Use video encoding for camera observations if supported by the selected LeRobot v3 workflow.
- Record episode boundaries, frame indices, timestamps, task/episode identifiers, and the required metadata using the appropriate v3 schema.
- Do not invent nonstandard feature names or metadata fields without documenting them.
- Validate the exported dataset by reopening it with the corresponding LeRobot v3 reader/API.

### 5.2 Required data fields

Include the following information wherever the LeRobot v3 schema supports it:

- `observation.images.*`: RGB camera observations, with a fixed resolution and documented camera configuration.
- `observation.state`: robot state, including joint positions and/or end-effector pose, plus gripper state as appropriate.
- `action`: the actual action commands sent to the robot controller.
- `timestamp`: frame/step timestamps following the dataset's timing conventions.
- `episode_index`, `frame_index`, and task identifiers according to the official schema.

The exact feature names, shapes, dtypes, video configuration, and metadata fields must be consistent across episodes.

Also preserve object-level ground truth and interaction events, including:
- Stable object asset keys.
- Object poses and velocities.
- Receptacle pose.
- Intended object order and actual grasp/place order.
- Grasp and placement success events.
- Episode success/failure and failure reasons.
- Random seeds and initial scene configuration.

If object-level GT and event logs are not appropriate standard LeRobot features, store them as documented auxiliary files keyed by episode index. Do not break compatibility with standard LeRobot v3 readers.

### 5.3 Paired Context–Query representation

Each Context episode and Query episode must be valid LeRobot v3 episodes.

Create a separate pair index, for example `pairs.jsonl` or another documented metadata file, linking:
- `pair_id`
- `context_episode_index`
- `query_episode_index`
- `intended_object_order`
- Context and Query scene seeds
- Validation status

The pair index is auxiliary metadata; the underlying demonstration dataset must remain readable using the standard LeRobot v3 API.

The Context and Query episodes must share the same intended object order but have independently initialized scene layouts. Store the intended order separately from the actual order observed during execution.

The Query expert actions and ground-truth trajectory are for offline evaluation only. They must not be exposed as part of the model's Query input.

### 5.4 Validation requirements

After generating the first paired episodes:

1. Load the dataset using the official LeRobot v3 reader/API.
2. Verify that all episodes, feature definitions, timestamps, and video streams can be read.
3. Check that video frames align with action/state timestamps within the documented tolerance.
4. Verify that Context–Query links point to valid episode indices.
5. Confirm that intended and actual object orders agree for every episode labeled successful.
6. Confirm that unsuccessful episodes are not silently labeled successful.
7. Provide a small inspection script that loads a pair and visualizes its Context and Query videos alongside the corresponding action/state information.

The acceptance criterion is not merely that files exist: the generated dataset must be readable and structurally valid under the installed LeRobot v3 implementation.

### 5.5 Compatibility and implementation strategy

Prioritize a minimal, standards-compliant LeRobot v3 export before implementing large-scale collection.

If the installed LeRobot version lacks a required feature:
- Identify the exact limitation.
- Check whether the official v3 API supports an alternative.
- Keep auxiliary metadata separate when necessary.
- Document any deviation explicitly instead of silently reverting to a custom format.

Do not implement a custom replacement for the LeRobot writer unless the official API cannot support the required workflow.

## 6. Important: Record Interaction Events

Implement reliable event detection instead of inferring the object order from video filenames or manually assigned labels.

At minimum, record events such as:

- `grasp_attempt`
- `grasp_success`
- `object_lifted`
- `place_attempt`
- `place_success`
- `episode_success`
- `episode_failure`

Each event should include a simulation timestamp or step index and the relevant object key.

A grasp should only count as successful when the object is actually held and responds to the robot's movement. A placement should only count as successful after the object is released and its state is verified within the receptacle.

The metadata's intended order and the actual order should be stored separately. A failed demonstration must not be silently labeled as successful.

## 7. Experimental Controls

The initial version should isolate visual/action Context from language.

Keep the following fixed whenever possible:
- Robot embodiment.
- Gripper.
- Controller and action representation.
- Camera viewpoint and image resolution.
- Lighting and rendering settings.
- Object asset identities within each pair.
- Object count.
- Receptacle type.
- Success thresholds.

Randomize scene geometry only to the extent needed to produce useful paired data.

The order should vary across pairs, with Query order determined by the paired Context. This prevents the generator from always assigning the same order to each object type.

Avoid using duplicated or indistinguishable objects in the initial prototype because they make visual order inference ambiguous.

The first dataset should measure transfer of an observed object sequence across different scene layouts. It does not need to solve novel language instructions or novel object categories.

## 8. Paired Evaluation Conditions

Design the metadata so the same Query can later be evaluated under these conditions:

1. **Full Context:** complete Context RGB sequence and time-aligned actions.
2. **Endpoint-only Context:** only the initial and final Context images.
3. **No Context:** the Query is evaluated without a demonstration.
4. **Mismatched Context:** a demonstration with the same or similar object set but a different object order.

These conditions can initially be implemented as metadata and lightweight data-loading utilities; do not build a VLA inference framework.

Do not assume Full Context must always outperform the baselines. The purpose is to create controlled data that allows this question to be tested.

## 9. Implementation Plan

Work incrementally.

### Phase 1: Repository audit

Inspect the current code and identify:
- How `PickCube-v1` is implemented.
- How its motion-planning demonstration is generated.
- How actors and object assets are loaded.
- How offscreen RGB rendering and recording work.
- How observation/action data are saved.
- Which parts can be reused without duplicating infrastructure.

Summarize the relevant files and proposed implementation before modifying many files.

### Phase 2: Minimal multi-object task

Build one scene with three objects and one receptacle.

Demonstrate one physically successful sequence:
- Pick object A and place it inside.
- Pick object B and place it inside.
- Pick object C and place it inside.

Record the episode and verify every placement using simulator state.

Do not proceed to large-scale data generation until this works reliably.

### Phase 3: Paired episodes

Generate a Context and a Query with:
- The same object identities.
- Different initial object positions.
- Potentially different receptacle positions.
- A shared intended object order.

Verify the pair automatically using metadata and interaction events.

### Phase 4: Data collection and validation

Save synchronized frames, actions, robot states, object states, events, video, and metadata.

Produce a short summary report containing:
- Number of pairs attempted.
- Number and percentage of successful Context episodes.
- Number and percentage of successful Query episodes.
- Grasp success rate.
- Placement success rate.
- Failure reasons.
- Rendering resolution and actual dataset size.
- Time required to generate one pair.

Provide a dataset inspection script that can replay or visualize a selected episode and display its recorded object order and success status.

### Phase 5: Generalization and extensibility

Only after the baseline works:
- Add configurable object asset lists.
- Add optional YCB object support.
- Add multiple receptacle geometries.
- Add controlled changes in object orientation.
- Add train/validation/test split generation.
- Add optional multi-camera, depth, and segmentation recording.

Do not make these extensions a prerequisite for completing the initial version.

## 10. Acceptance Criteria

The implementation is considered complete when all the following are true:

1. A documented command creates a multi-object Pick-and-Place scene without requiring a GUI.
2. A scripted robot policy successfully picks and places all three objects using physical robot actions.
3. The actual grasp-and-place sequence is automatically recorded and verified.
4. The generator produces at least one valid Context–Query pair with the required shared object order and independently randomized layouts.
5. RGB frames, actions, timestamps, robot states, object GT, event logs, and episode metadata are saved.
6. A script can reload the metadata and verify that the intended order, actual order, and success labels are consistent.
7. The generated episode can be replayed or inspected offline.
8. The README documents installation assumptions, commands, data schema, limitations, and failure troubleshooting.

Please implement the actual code and run small tests. Do not stop after producing a design document.

If a component does not work, isolate the failure and report the exact error and a minimal reproducible command. Do not claim success if only scene initialization or random-action rendering has been tested.

After implementation, provide:
- A summary of files created or modified.
- The exact command for generating a small dataset.
- The exact command for visualizing or replaying one pair.
- The directory containing generated data.
- Test results and any known limitations.

## 11. Response Language

**Please communicate with me in Chinese throughout the entire process.** This includes your initial implementation plan, code explanations, progress summaries, debugging questions, test results, final report, and any README or usage instructions you generate for me.

Code identifiers, API names, file paths, command-line arguments, and standard schema fields should remain in their conventional technical form. Explanations and documentation intended for me should be written in Chinese.