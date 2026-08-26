# AS2-W 24–40 s visual acceptance reference

The north-star reference for Go2W trick work is the [Unitree AS2-W video](https://www.youtube.com/watch?v=Xha2TJ_1C6Q), **24.00–40.00 s**.  It is an evaluation reference only: no extracted frame, pose, phase, or trajectory may enter an actor observation or reward as an imitation target.

## Preserved source material

- Exact clip: `artifacts/reference/as2w_xha2tj_1c6q/as2w-24-40.mp4`
- Dense archive: `artifacts/reference/as2w_xha2tj_1c6q/frames_50fps/frame_0001.jpg` through `frame_0800.jpg`
- Frame mapping: reference timestamp = `24.00 + (frame_number - 1) / 50` seconds.

The source is 50 fps.  Keep this archive outside Git because it is large, and use dense frames rather than a sparse overview for high-speed motions.

## Observed requirements

1. Frames 0201–0239 (about 28.00–28.76 s) show the clearest high-speed contact pivot.  The contact pair is the laterally separated front or rear pair, forming a balancing axle; it is not a same-side bicycle pair driving across the ground.
2. The support-pair midpoint remains local while the torso turns rapidly.  The apparent turn is consistent with a world-down turn of the tall support, and is the intended command semantics, but a monocular, moving showreel shot cannot measure a 3-D axis or an exact rate.  The roughly **0.76 s / 8 rad/s** estimate is therefore only a visual speed target, never a training label.
3. The non-support legs visibly move throughout the pivot.  A rigid, cylinder-like posture is not visually acceptable even if scalar rewards are high.
4. The robot returns to ordinary four-wheel support around frames 0261 onward.  A spin that ends in a body/hip collision, crouch, or uncontrolled drift is a failure.
5. Later airborne events must show a real launch, a signed rotation in the commanded direction, and a controlled return.  They must not be judged from reward alone or from a single still image.

## Frame-by-frame motion notes

The video is a showreel: it contains camera cuts and changing terrain, so it
cannot supply centimetre-scale world trajectories.  These are deliberately
visual observations, not invented labels for policy training.

| Reference range | What is visible | Environment consequence |
| --- | --- | --- |
| 26.0–27.3 s, frames 0101–0165 | Ordinary four-wheel motion on level ground. | Start every spin attempt from the regular four-wheel reset.  No hidden two-wheel reset pose. |
| 27.3–28.0 s, frames 0166–0200 | A continuous, active rise: the legs extend and reshape while contact is reduced to a lateral front/rear pair. | The policy must discover the rise from the normal reset.  Do not prescribe its joint pose or a phase trajectory. |
| 28.0–29.2 s, frames 0201–0260 | A laterally separated front/rear wheel pair is the support axle.  The support is tall, turns quickly around a local centre, and then returns to four wheels.  The clearest turning interval is frames 0201–0239. | Fixed front/rear spin: exact support pair, commanded signed world-down turn, support-midpoint drift check, and normal-wheel recovery.  Same-side supports are not assigned this motion without contrary visual evidence. |
| about 30.2–31.1 s, frames 0311–0355 | First aerial shot: a genuine launch, wheel-free inverted flight, visible limb reshaping, then wheel-first recovery. | It is valid evidence for the aerial task's launch/flight/landing structure, but the camera does not let us label a body-frame sign from pixels alone. |
| about 32.2–32.9 s, frames 0411–0445 | Second aerial shot with the same physical structure, from another camera/terrain view. | Keep front/back as distinct signed commands in simulation, but validate their signs from simulator state as well as video. |
| about 33.5–34.2 s, frames 0476–0510 | Third rapid aerial shot, again with all wheels clear and no body-supported landing. | Do not reward a ground scrape, a body bounce, or a wheel graze as a flip. |
| 36.0–40.0 s, frames 0601–0800 | A second dynamic sequence has substantial coordinated limb motion and alternates between a compact four-wheel normal form and tall two-wheel forms.  In the normal form the trunk stays level and visibly clear of the wheel axle, all four wheels remain low, and the wheel axes are folded onto one transverse line under the body; it is visually unlike a frozen four-bar linkage, a splayed crouch, a belly-low chassis, or bicycle-like translation.  The clip ends mid-maneuver. | A nonzero normal-rate branch must retain four wheel contacts, level trunk, measured trunk-to-wheel clearance, compact common axle, local all-wheel centroid, and signed world-down rotation.  Reward neither a joint pose nor a hidden phase/reference trajectory.  Acceptance needs full video, support-centre drift, and contact audit; named one-hots retain their separately measured two-wheel supports. |

Across both tasks, the key visual distinction is **coordinated moving legs in a
compact envelope**, not either extreme: neither a rigid cylinder nor wide,
collision-driven flailing is acceptable.  This motivates compliant
torque-limited actuators and a measured safety envelope, rather than a
time-indexed joint target.

## Required validation for every claimed skill

For each checkpoint under review:

1. Run fixed-command, headless metrics for every one-hot mode, including non-wheel/body contact and support-centre drift.
2. Record deterministic videos from the ordinary four-wheel reset.  Use the same command duration and a camera view that makes the support pair and translation visible.
3. Extract 50-fps frames (or preserve the native frame rate if higher) around the fast turn/flight, then compare them against this archive.  A contact pivot must visibly remain local; aerials must visibly rotate in the commanded sign and land.
4. Do not call a mode learned if a population mean, reward, or isolated seed passes while the video has mode collapse, body support, bicycle-like translation, or a rigid-leg bounce.

Current command design remains compact: one five-way one-hot plus spin-rate.
Every non-idle mode interprets a nonzero signed rate as a dynamic pivot request.
The normal branch uses the visible four-wheel common-axle outcome; named
branches use their named measured supports.  The environment never injects a
joint pose, phase, or reference path.  Gravity targets, contacts, axle
geometry, and support-centre checks are task-side outcome criteria, not extra
actor observations or a reference trajectory.
