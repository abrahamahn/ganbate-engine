//! Physics in the web runtime: the cooked world's `rigid_body` and `collider` components,
//! simulated by `ganbate-physics`, moving the world's instances and lamps.
//!
//! When a world with physics components loads, its bodies are built once and kept as the
//! authored starting state, or, when they cannot be built, the reason
//! (`ganbate_rt_physics_error`). `ganbate_rt_physics_start` copies that state into a live simulation,
//! `ganbate_rt_physics_step` advances it and moves what each moving body carries (the instances and
//! lamps of its entity and of the descendants that have no body of their own), only for the
//! bodies that moved, and `ganbate_rt_physics_stop` drops it and puts everything back where it was
//! cooked, so the editor can press play and stop as often as it likes.

use std::cell::RefCell;

use ganbate_engine::world::{EntityTable, World};
use ganbate_physics::scene::{self, ScenePhysics};
use ganbate_physics::{BodyKind, Iso, Settings, Vec3};

use crate::say;

/// What one body moves: world instances with their cooked matrices, and world lamps with their
/// cooked positions.
#[derive(Default)]
struct Carried {
    instances: Vec<(u32, [f32; 16])>,
    lamps: Vec<(u32, Vec3)>,
}

struct Live {
    scene: ScenePhysics,
    /// Each body's pose when what it carries was last moved.
    shown: Vec<Option<Iso>>,
}

struct WebPhysics {
    initial: ScenePhysics,
    live: Option<Live>,
    /// What each body (in `initial.bodies` order) moves: nothing for a static one.
    carried: Vec<Carried>,
}

enum State {
    /// The world has no physics components.
    Absent,
    /// The world's physics components could not be built: why.
    Failed(String),
    Ready(Box<WebPhysics>),
}

thread_local! {
    static PHYSICS: RefCell<State> = const { RefCell::new(State::Absent) };
}

/// Moves instances and lamps to the places given. Returns 1, or 0 after logging why not.
fn show(places: &Carried) -> u32 {
    let mut ok = 1;
    if !places.instances.is_empty() {
        ok = crate::runtime::move_world_instances(&places.instances);
    }
    for &(lamp, p) in &places.lamps {
        ok &= crate::runtime::ganbate_rt_set_world_light(lamp, p.x, p.y, p.z);
    }
    ok
}

impl WebPhysics {
    /// Where the bodies that moved since `live.shown` carry their instances and lamps now.
    fn moves(&self, live: &mut Live) -> Carried {
        let mut out = Carried::default();
        for i in live.scene.moved_since(&mut live.shown) {
            let (Some(c), Some(motion)) = (self.carried.get(i), live.scene.motion(i)) else {
                continue;
            };
            for (inst, cooked) in &c.instances {
                if let Some(m) = live.scene.instance_matrix(i, cooked) {
                    out.instances.push((*inst, m));
                }
            }
            out.lamps
                .extend(c.lamps.iter().map(|&(l, p)| (l, motion.point(p))));
        }
        out
    }

    /// Where the instances and lamps of the bodies `live` moved were cooked.
    fn restore(&self, live: &Live) -> Carried {
        let mut out = Carried::default();
        let built = self.initial.poses();
        for ((c, shown), built) in self.carried.iter().zip(&live.shown).zip(&built) {
            if shown != built {
                out.instances.extend_from_slice(&c.instances);
                out.lamps.extend_from_slice(&c.lamps);
            }
        }
        out
    }
}

/// Builds the authored physics of a freshly decoded world (or forgets the last world's).
pub(crate) fn prepare(world: &World, table: &EntityTable) {
    let wanted = table.data.contains("\"rigid_body\"") || table.data.contains("\"collider\"");
    let state = if !wanted {
        State::Absent
    } else {
        match scene::build(world, table, Settings::default()) {
            Ok(initial) => {
                let carried = initial
                    .bodies
                    .iter()
                    .map(|b| {
                        let mut c = Carried::default();
                        if initial.world.body_kind(b.body) == Some(BodyKind::Static) {
                            return c;
                        }
                        for rec in b
                            .carries
                            .iter()
                            .filter_map(|&r| table.entities.get(r as usize))
                        {
                            for inst in rec.instances.clone() {
                                if let Some(wi) = world.instances.get(inst as usize) {
                                    c.instances.push((inst, wi.transform.0));
                                }
                            }
                            for lamp in rec.lights.clone() {
                                if let Some(l) = world.lights.get(lamp as usize) {
                                    let p = l.position;
                                    c.lamps.push((lamp, Vec3::new(p.0, p.1, p.2)));
                                }
                            }
                        }
                        c
                    })
                    .collect();
                State::Ready(Box::new(WebPhysics {
                    initial,
                    live: None,
                    carried,
                }))
            }
            Err(e) => {
                say(1, &format!("physics: {e}"));
                State::Failed(e)
            }
        }
    };
    PHYSICS.with(|cell| *cell.borrow_mut() = state);
}

/// Starts (or restarts, after putting back what the last run moved) the simulation from the
/// authored state. Returns the number of bodies plus one, or 0 when the world has no physics or
/// it could not be built (`ganbate_rt_physics_error` says why).
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_physics_start() -> u32 {
    let (started, restore) = PHYSICS.with(|cell| {
        let mut state = cell.borrow_mut();
        let State::Ready(p) = &mut *state else {
            return (0, Carried::default());
        };
        let restore = p
            .live
            .take()
            .map(|live| p.restore(&live))
            .unwrap_or_default();
        p.live = Some(Live {
            shown: p.initial.poses(),
            scene: p.initial.clone(),
        });
        (p.initial.bodies.len() as u32 + 1, restore)
    });
    show(&restore);
    started
}

/// Advances the simulation `steps` times by `dt` seconds each, then moves the instances and lamps
/// of the bodies that moved: sleeping and still bodies cost nothing, and several steps in one
/// call move things once. At most [`scene::MAX_STEPS_PER_CALL`] (64) steps run in one call, each
/// at most [`scene::MAX_STEP`] (1/15 s) long ([`ScenePhysics::advance`]), so a host that passes a
/// frame count or milliseconds as `steps` cannot freeze the page. Returns 1, or 0 when it is not
/// running.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_physics_step(dt: f32, steps: u32) -> u32 {
    let moves = PHYSICS.with(|cell| {
        let mut state = cell.borrow_mut();
        let State::Ready(p) = &mut *state else {
            return None;
        };
        let mut live = p.live.take()?;
        live.scene.advance(dt, steps);
        let moves = p.moves(&mut live);
        p.live = Some(live);
        Some(moves)
    });
    moves.map_or(0, |m| show(&m))
}

/// Stops the simulation and puts every moved instance and lamp back where it was cooked.
/// Returns 1.
#[unsafe(no_mangle)]
pub extern "C" fn ganbate_rt_physics_stop() -> u32 {
    let restore = PHYSICS.with(|cell| {
        let mut state = cell.borrow_mut();
        let State::Ready(p) = &mut *state else {
            return None;
        };
        let live = p.live.take()?;
        Some(p.restore(&live))
    });
    if let Some(r) = restore {
        show(&r);
    }
    1
}

/// Writes the live simulation's state hash (8 bytes, little-endian) at `out`: equal hashes after
/// the same steps show the browser computed exactly what a native build computes. Returns 1, or
/// 0 when it is not running.
///
/// # Safety
/// `out` must address 8 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_physics_hash(out: *mut u8) -> u32 {
    PHYSICS.with(|cell| {
        let state = cell.borrow();
        let State::Ready(p) = &*state else {
            return 0;
        };
        let Some(live) = &p.live else {
            return 0;
        };
        let bytes = live.scene.world.state_hash().to_le_bytes();
        // SAFETY: the caller passes 8 writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, 8) };
        1
    })
}

/// Writes why the loaded world's physics components could not be built (UTF-8, naming the
/// entity) to `out` when it fits in `cap` bytes; returns its length (0 when they were built or
/// the world has none).
///
/// # Safety
/// `out` must address `cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ganbate_rt_physics_error(out: *mut u8, cap: usize) -> u32 {
    PHYSICS.with(|cell| match &*cell.borrow() {
        // SAFETY: forwarded from the caller.
        State::Failed(e) => unsafe { crate::give(e.as_bytes(), out, cap) },
        _ => 0,
    })
}
