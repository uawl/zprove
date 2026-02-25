use crate::transition::{AccessDomain, AccessKind, AccessRecord, InstructionTransitionStatement};
use p3_field::PrimeField32;
use p3_mersenne_31::{Mersenne31, Poseidon2Mersenne31};
use p3_symmetric::Permutation;
use rand::SeedableRng;
use rand_xoshiro::Xoroshiro128Plus;
use revm::bytecode::opcode;
use std::collections::BTreeMap;
use std::sync::OnceLock;

const POSEIDON_WIDTH: usize = 16;

fn poseidon2_perm() -> &'static Poseidon2Mersenne31<POSEIDON_WIDTH> {
  static POSEIDON: OnceLock<Poseidon2Mersenne31<POSEIDON_WIDTH>> = OnceLock::new();
  POSEIDON.get_or_init(|| {
    let mut rng = Xoroshiro128Plus::seed_from_u64(1);
    Poseidon2Mersenne31::new_from_rng_128(&mut rng)
  })
}

fn poseidon2_hash_bytes(input: &[u8]) -> [u8; 32] {
  assert!(
    input.len() <= 64,
    "Poseidon2 preimage too long for current hasher"
  );

  let mut state = Mersenne31::new_array([0u32; POSEIDON_WIDTH]);
  for (i, chunk) in input.chunks(4).enumerate() {
    let mut limb = [0u8; 4];
    limb[..chunk.len()].copy_from_slice(chunk);
    let value = u32::from_le_bytes(limb) & 0x7fff_ffff;
    state[i] = Mersenne31::new(value);
  }
  state[POSEIDON_WIDTH - 2] = Mersenne31::new(input.len() as u32);
  state[POSEIDON_WIDTH - 1] = Mersenne31::new(1);

  poseidon2_perm().permute_mut(&mut state);

  let mut out = [0u8; 32];
  for i in 0..8 {
    out[i * 4..(i + 1) * 4].copy_from_slice(&state[i].as_canonical_u32().to_le_bytes());
  }
  out
}

fn u256_to_u64_checked(word: &[u8; 32]) -> Option<u64> {
  if word[..24].iter().any(|&b| b != 0) {
    return None;
  }
  let mut out = [0u8; 8];
  out.copy_from_slice(&word[24..32]);
  Some(u64::from_be_bytes(out))
}

pub fn verify_memory_semantics(statement: &InstructionTransitionStatement) -> bool {
  let memory_accesses: Vec<&AccessRecord> = statement
    .accesses
    .iter()
    .filter(|access| access.domain == AccessDomain::Memory)
    .collect();

  match statement.opcode {
    opcode::MLOAD => {
      if memory_accesses.len() != 1 {
        return false;
      }
      let access = memory_accesses[0];
      let Some(offset) = u256_to_u64_checked(&statement.s_i.stack[0]) else {
        return false;
      };
      if access.kind != AccessKind::Read || access.addr != offset || access.width != 32 {
        return false;
      }
      match (access.value_before, access.value_after) {
        (Some(before), None) => statement.s_next.stack[0] == before,
        _ => false,
      }
    }
    opcode::MSTORE => {
      if memory_accesses.len() != 1 {
        return false;
      }
      let access = memory_accesses[0];
      let Some(offset) = u256_to_u64_checked(&statement.s_i.stack[0]) else {
        return false;
      };
      if !matches!(access.kind, AccessKind::Write | AccessKind::ReadWrite)
        || access.addr != offset
        || access.width != 32
      {
        return false;
      }
      match access.value_after {
        Some(after) => after == statement.s_i.stack[1],
        None => false,
      }
    }
    opcode::MSTORE8 => {
      if memory_accesses.len() != 1 {
        return false;
      }
      let access = memory_accesses[0];
      let Some(offset) = u256_to_u64_checked(&statement.s_i.stack[0]) else {
        return false;
      };

      let aligned_base = offset & !31;
      let byte_index = (offset & 31) as usize;

      if !matches!(access.kind, AccessKind::Write | AccessKind::ReadWrite)
        || access.addr != aligned_base
        || access.width != 32
      {
        return false;
      }
      match (access.value_before, access.value_after) {
        (Some(before), Some(after)) => {
          let mut expected = before;
          expected[byte_index] = statement.s_i.stack[1][31];
          after == expected
        }
        _ => false,
      }
    }
    _ => memory_accesses.is_empty(),
  }
}

fn verify_memory_access_shape(access: &AccessRecord) -> bool {
  if access.width == 0 {
    return false;
  }

  match access.kind {
    AccessKind::Read => {
      access.value_before.is_some()
        && access.value_after.is_none()
        && access.merkle_path_after.is_empty()
    }
    AccessKind::Write => {
      access.value_before.is_some()
        && access.value_after.is_some()
        && access.merkle_path_before.len() == access.merkle_path_after.len()
    }
    AccessKind::ReadWrite => {
      access.value_before.is_some()
        && access.value_after.is_some()
        && access.merkle_path_before.len() == access.merkle_path_after.len()
    }
  }
}

fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
  let mut preimage = [0u8; 64];
  preimage[..32].copy_from_slice(left);
  preimage[32..].copy_from_slice(right);
  poseidon2_hash_bytes(&preimage)
}

fn memory_leaf_hash(addr: u64, width: u32, value: &[u8; 32]) -> [u8; 32] {
  let mut preimage = [0u8; 45];
  preimage[0] = 1;
  preimage[1..9].copy_from_slice(&addr.to_le_bytes());
  preimage[9..13].copy_from_slice(&width.to_le_bytes());
  preimage[13..45].copy_from_slice(value);
  poseidon2_hash_bytes(&preimage)
}

pub fn compute_memory_root(addr: u64, width: u32, value: &[u8; 32], path: &[[u8; 32]]) -> [u8; 32] {
  let mut node = memory_leaf_hash(addr, width, value);
  for (level, sibling) in path.iter().enumerate() {
    let bit = (addr >> level) & 1;
    node = if bit == 0 {
      hash_pair(&node, sibling)
    } else {
      hash_pair(sibling, &node)
    };
  }
  node
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CqRw {
  Read,
  Write,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CqMemoryEvent {
  pub addr: u64,
  pub step: u64,
  pub value: [u8; 32],
  pub rw: CqRw,
  pub width: u32,
}

pub fn build_memory_cq_events(
  statement: &InstructionTransitionStatement,
) -> Option<Vec<CqMemoryEvent>> {
  let mut out = Vec::new();

  for access in &statement.accesses {
    if access.domain != AccessDomain::Memory {
      continue;
    }

    match access.kind {
      AccessKind::Read => {
        let before = access.value_before?;
        out.push(CqMemoryEvent {
          addr: access.addr,
          step: access.rw_counter,
          value: before,
          rw: CqRw::Read,
          width: access.width,
        });
      }
      AccessKind::Write => {
        let after = access.value_after?;
        out.push(CqMemoryEvent {
          addr: access.addr,
          step: access.rw_counter,
          value: after,
          rw: CqRw::Write,
          width: access.width,
        });
      }
      AccessKind::ReadWrite => {
        let before = access.value_before?;
        let after = access.value_after?;
        out.push(CqMemoryEvent {
          addr: access.addr,
          step: access.rw_counter,
          value: before,
          rw: CqRw::Read,
          width: access.width,
        });
        out.push(CqMemoryEvent {
          addr: access.addr,
          step: access.rw_counter,
          value: after,
          rw: CqRw::Write,
          width: access.width,
        });
      }
    }
  }

  Some(out)
}

pub fn verify_memory_cq_constraints(statement: &InstructionTransitionStatement) -> bool {
  let memory_accesses: Vec<&AccessRecord> = statement
    .accesses
    .iter()
    .filter(|access| access.domain == AccessDomain::Memory)
    .collect();

  if memory_accesses.is_empty() {
    return true;
  }

  let Some(events) = build_memory_cq_events(statement) else {
    return false;
  };

  if events.is_empty() {
    return false;
  }

  let mut last_step = None;
  for event in &events {
    if event.width == 0 {
      return false;
    }
    if let Some(prev) = last_step
      && event.step < prev
    {
      return false;
    }
    last_step = Some(event.step);
  }

  let mut last_value_by_cell: BTreeMap<(u64, u32), [u8; 32]> = BTreeMap::new();
  for access in memory_accesses {
    let key = (access.addr, access.width);

    match access.kind {
      AccessKind::Read => {
        let Some(observed) = access.value_before else {
          return false;
        };
        if let Some(last) = last_value_by_cell.get(&key) {
          if *last != observed {
            return false;
          }
        } else {
          last_value_by_cell.insert(key, observed);
        }
      }
      AccessKind::Write => {
        let Some(before) = access.value_before else {
          return false;
        };
        if let Some(last) = last_value_by_cell.get(&key) {
          if *last != before {
            return false;
          }
        } else {
          last_value_by_cell.insert(key, before);
        }

        let Some(after) = access.value_after else {
          return false;
        };
        last_value_by_cell.insert(key, after);
      }
      AccessKind::ReadWrite => {
        let Some(before) = access.value_before else {
          return false;
        };
        if let Some(last) = last_value_by_cell.get(&key) {
          if *last != before {
            return false;
          }
        } else {
          last_value_by_cell.insert(key, before);
        }

        let Some(after) = access.value_after else {
          return false;
        };
        last_value_by_cell.insert(key, after);
      }
    }
  }

  true
}

pub fn verify_memory_access_commitments(statement: &InstructionTransitionStatement) -> bool {
  let memory_accesses: Vec<&AccessRecord> = statement
    .accesses
    .iter()
    .filter(|access| access.domain == AccessDomain::Memory)
    .collect();

  if memory_accesses.is_empty() {
    return statement.s_i.memory_root == statement.s_next.memory_root;
  }

  if !verify_memory_cq_constraints(statement) {
    return false;
  }

  let mut current_root = statement.s_i.memory_root;
  let mut last_rw_counter = None;

  for access in memory_accesses {
    if !verify_memory_access_shape(access) {
      return false;
    }

    if let Some(last) = last_rw_counter
      && access.rw_counter <= last
    {
      return false;
    }
    last_rw_counter = Some(access.rw_counter);

    let before_value = match access.value_before {
      Some(value) => value,
      None => return false,
    };
    let root_before = compute_memory_root(
      access.addr,
      access.width,
      &before_value,
      &access.merkle_path_before,
    );
    if root_before != current_root {
      return false;
    }

    if matches!(access.kind, AccessKind::Write | AccessKind::ReadWrite) {
      let after_value = match access.value_after {
        Some(value) => value,
        None => return false,
      };
      current_root = compute_memory_root(
        access.addr,
        access.width,
        &after_value,
        &access.merkle_path_after,
      );
    }
  }

  current_root == statement.s_next.memory_root
}
