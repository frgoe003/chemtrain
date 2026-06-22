#!/usr/bin/env python3
# Copyright 2026 Multiscale Modeling of Fluid Materials, TU Munich
# SPDX-License-Identifier: Apache-2.0
"""Run the end-to-end chemtrain communication regression."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import ase.io
import numpy as onp


MAX_ATOMIC_ENERGY_ERROR_EV = 1.0e-4
MAX_FORCE_ERROR_EV_PER_ANGSTROM = 5.0e-3
MAX_POSITION_ERROR_ANGSTROM = 1.0e-4
MAX_VELOCITY_ERROR_ANGSTROM_PER_PS = 1.0e-3
MAX_DRIFT_ERROR_EV_PER_ATOM = 2.0e-5
MACE_ENERGY_ERROR_EV_PER_ATOM = 1.0e-3
MACE_FORCE_ERROR_EV_PER_ANGSTROM = 5.0e-2
MODEL_CUTOFF_ANGSTROM = 5.0


def run_command(
    name: str,
    command: list[str],
    *,
    working_directory: Path,
    output_directory: Path,
    environment: dict[str, str],
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run an external command and save its combined screen output."""
    print(f"\n[{name}] {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=working_directory,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output_directory / f"{name}.screen").write_text(completed.stdout)
    if expect_success and completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {completed.returncode}; see "
            f"{output_directory / f'{name}.screen'}"
        )
    if not expect_success and completed.returncode == 0:
        raise AssertionError(f"{name} unexpectedly succeeded")
    return completed


def read_lammps_dump(path: Path) -> list[dict[str, onp.ndarray]]:
    """Read a LAMMPS custom dump and sort every frame by atom ID."""
    frames: list[dict[str, onp.ndarray]] = []
    lines = path.read_text().splitlines()
    index = 0
    while index < len(lines):
        if lines[index] != "ITEM: TIMESTEP":
            index += 1
            continue
        step = int(lines[index + 1])
        count = int(lines[index + 3])
        header_index = index + 8
        header = lines[header_index].split()
        if header[:2] != ["ITEM:", "ATOMS"]:
            raise RuntimeError(f"Malformed atom header in {path}")
        columns = header[2:]
        data = onp.asarray(
            [
                [float(value) for value in line.split()]
                for line in lines[
                    header_index + 1 : header_index + 1 + count
                ]
            ]
        )
        column_index = {name: number for number, name in enumerate(columns)}
        data = data[onp.argsort(data[:, column_index["id"]].astype(int))]
        frame = {name: data[:, number] for name, number in column_index.items()}
        frame["step"] = onp.asarray(step)
        frames.append(frame)
        index = header_index + 1 + count
    if not frames:
        raise RuntimeError(f"No frames found in {path}")
    return frames


def read_potential_energies(text: str) -> onp.ndarray:
    """Collect potential energies from all LAMMPS thermo tables in order."""
    energies: list[float] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        header = line.split()
        if not header or header[0] != "Step" or "PotEng" not in header:
            continue
        energy_column = header.index("PotEng")
        for row in lines[index + 1 :]:
            fields = row.split()
            if len(fields) != len(header):
                break
            try:
                energies.append(float(fields[energy_column]))
            except ValueError:
                break
    if not energies:
        raise RuntimeError("No LAMMPS potential-energy table was found")
    return onp.asarray(energies)


def compare_lammps_runs(
    case: str,
    reference_dump: Path,
    communication_dump: Path,
    reference_screen: str,
    communication_screen: str,
) -> dict[str, float]:
    """Compare atom-aligned predictions and trajectory observables."""
    reference = read_lammps_dump(reference_dump)
    communication = read_lammps_dump(communication_dump)
    if len(reference) != len(communication):
        raise AssertionError(
            f"{case}: frame counts differ: {len(reference)} and "
            f"{len(communication)}"
        )

    position_errors = []
    velocity_errors = []
    force_errors = []
    atomic_energy_errors = []
    for frame_index, (expected, actual) in enumerate(
        zip(reference, communication, strict=True)
    ):
        for exact_column in ("id", "type", "step"):
            onp.testing.assert_array_equal(
                actual[exact_column],
                expected[exact_column],
                err_msg=f"{case}, frame {frame_index}: {exact_column} differs",
            )
        # Trajectories use unwrapped coordinates, while static predictions use
        # wrapped coordinates.
        position_columns = (
            ("xu", "yu", "zu") if "xu" in expected else ("x", "y", "z")
        )
        position_errors.append(
            onp.column_stack(
                [actual[name] - expected[name] for name in position_columns]
            )
        )
        if all(name in expected for name in ("vx", "vy", "vz")):
            velocity_errors.append(
                onp.column_stack(
                    [actual[name] - expected[name] for name in ("vx", "vy", "vz")]
                )
            )
        force_errors.append(
            onp.column_stack(
                [actual[name] - expected[name] for name in ("fx", "fy", "fz")]
            )
        )
        if "c_atom_energy" not in expected or "c_atom_energy" not in actual:
            raise AssertionError(
                f"{case}, frame {frame_index}: per-atom energy is missing"
            )
        atomic_energy_errors.append(
            actual["c_atom_energy"] - expected["c_atom_energy"]
        )

    reference_energy = read_potential_energies(reference_screen)
    communication_energy = read_potential_energies(communication_screen)
    if reference_energy.shape != communication_energy.shape:
        raise AssertionError(
            f"{case}: thermo energy shapes differ: {reference_energy.shape} "
            f"and {communication_energy.shape}"
        )
    atoms = len(reference[0]["id"])
    total_energy_delta = (
        onp.abs(communication_energy - reference_energy) / atoms
    )
    total_energy_frame = int(onp.argmax(total_energy_delta))
    total_energy_error = float(total_energy_delta[total_energy_frame])
    all_atomic_energy_error_array = onp.abs(onp.asarray(atomic_energy_errors))
    # Once two NVE trajectories have advanced, their coordinates are no longer
    # identical and atomic energies are not like-for-like model predictions.
    # The initial trajectory frame and every static/rerun frame evaluate the
    # same coordinates and therefore support the strict per-atom comparison.
    atomic_energy_error_array = (
        all_atomic_energy_error_array[:1]
        if velocity_errors
        else all_atomic_energy_error_array
    )
    atomic_energy_index = onp.unravel_index(
        onp.argmax(atomic_energy_error_array), atomic_energy_error_array.shape
    )
    atomic_energy_error = float(atomic_energy_error_array[atomic_energy_index])
    force_error_array = onp.abs(onp.asarray(force_errors))
    force_index = onp.unravel_index(
        onp.argmax(force_error_array), force_error_array.shape
    )
    force_error = float(force_error_array[force_index])
    position_error_array = onp.abs(onp.asarray(position_errors))
    position_index = onp.unravel_index(
        onp.argmax(position_error_array), position_error_array.shape
    )
    position_error = float(position_error_array[position_index])
    if velocity_errors:
        velocity_error_array = onp.abs(onp.asarray(velocity_errors))
        velocity_index = onp.unravel_index(
            onp.argmax(velocity_error_array), velocity_error_array.shape
        )
        velocity_error = float(velocity_error_array[velocity_index])
        velocity_location = (
            f"frame {velocity_index[0]}, atom {velocity_index[1] + 1}, "
            f"component {velocity_index[2]}"
        )
    else:
        velocity_error = 0.0
        velocity_location = "static frame"
    reference_drift = reference_energy - reference_energy[0]
    communication_drift = communication_energy - communication_energy[0]
    drift_delta = onp.abs(communication_drift - reference_drift) / atoms
    drift_frame = int(onp.argmax(drift_delta))
    drift_error = float(drift_delta[drift_frame])

    limits = {
        "atomic_energy_error_ev": (
            atomic_energy_error,
            MAX_ATOMIC_ENERGY_ERROR_EV,
        ),
        "force_error_ev_per_angstrom": (
            force_error,
            MAX_FORCE_ERROR_EV_PER_ANGSTROM,
        ),
        "position_error_angstrom": (
            position_error,
            MAX_POSITION_ERROR_ANGSTROM,
        ),
        "velocity_error_angstrom_per_ps": (
            velocity_error,
            MAX_VELOCITY_ERROR_ANGSTROM_PER_PS,
        ),
        "drift_error_ev_per_atom": (
            drift_error,
            MAX_DRIFT_ERROR_EV_PER_ATOM,
        ),
    }
    locations = {
        "atomic_energy_error_ev": (
            f"frame {atomic_energy_index[0]}, atom ID "
            f"{int(reference[atomic_energy_index[0]]['id'][atomic_energy_index[1]])}"
        ),
        "force_error_ev_per_angstrom": (
            f"frame {force_index[0]}, atom {force_index[1] + 1}, "
            f"component {force_index[2]}"
        ),
        "position_error_angstrom": (
            f"frame {position_index[0]}, atom {position_index[1] + 1}, "
            f"component {position_index[2]}"
        ),
        "velocity_error_angstrom_per_ps": velocity_location,
        "drift_error_ev_per_atom": f"thermo row {drift_frame}",
    }
    for quantity, (measured, tolerance) in limits.items():
        if measured > tolerance:
            raise AssertionError(
                f"{case}: {quantity}={measured:.6g} exceeds "
                f"{tolerance:.6g} at {locations[quantity]}"
            )
    print(
        f"PASS {case}: max atomic dE={atomic_energy_error:.3e} eV, "
        f"total dE/atom={total_energy_error:.3e} eV, "
        f"max dF={force_error:.3e} eV/Angstrom",
        flush=True,
    )
    metrics = {name: measured for name, (measured, _) in limits.items()}
    # Retain the total-energy difference as a diagnostic. Pass/fail is based on
    # the unsummed atomic predictions, while trajectory drift remains checked
    # separately as a physical integration observable.
    metrics["total_energy_error_ev_per_atom"] = total_energy_error
    if velocity_errors:
        metrics["trajectory_atomic_energy_error_ev"] = float(
            onp.max(all_atomic_energy_error_array)
        )
    return metrics


def run_trajectory_case(
    case: str,
    atom_padding: float,
    edge_padding: float,
    *,
    args: argparse.Namespace,
    script_directory: Path,
    output_directory: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    """Run one complete padding scenario through both model variants."""
    outputs: dict[str, tuple[Path, subprocess.CompletedProcess[str]]] = {}
    for variant, communication in (("default", "off"), ("comm", "on")):
        name = f"{case}_{variant}"
        dump = output_directory / f"{name}.lammpstrj"
        log = output_directory / f"{name}.log"
        command = [
            *shlex.split(args.launcher),
            args.lmp,
            "-var",
            "model",
            str(args.model.resolve()),
            "-var",
            "comm",
            communication,
            "-var",
            "atom_padding",
            str(atom_padding),
            "-var",
            "edge_padding",
            str(edge_padding),
            "-var",
            "trajectory_dump",
            str(dump),
            "-log",
            str(log),
            "-in",
            "trajectory.lmp",
        ]
        outputs[variant] = (
            dump,
            run_command(
                name,
                command,
                working_directory=script_directory,
                output_directory=output_directory,
                environment=environment,
            ),
        )

    metrics = compare_lammps_runs(
        case,
        outputs["default"][0],
        outputs["comm"][0],
        outputs["default"][1].stdout,
        outputs["comm"][1].stdout,
    )
    statistics: dict[str, dict[str, int]] = {}
    for variant, (_, completed) in outputs.items():
        text = completed.stdout
        statistics[variant] = {}
        for label, key in (
            ("Initial compilations", "initial"),
            ("Atom recompilations", "atom"),
            ("Edge recompilations", "edge"),
        ):
            stable_key = {
                "Initial compilations": "initial_total",
                "Atom recompilations": "atom_total",
                "Edge recompilations": "edge_total",
            }[label]
            matches = re.findall(rf"JCN_STATS .*?{stable_key}=([0-9]+)", text)
            if not matches:
                raise RuntimeError(f"{case}/{variant}: missing {label} summary")
            # LAMMPS reports and resets pair statistics for every `run`
            # command. The largest per-run count records whether this case
            # exercised the requested compilation path.
            statistics[variant][key] = max(int(value) for value in matches)

        if statistics[variant]["initial"] < 2:
            raise AssertionError(
                f"{case}/{variant}: expected initial compilation on both ranks, got "
                f"{statistics[variant]['initial']} total"
            )
        if case == "normal_padding" and (
            statistics[variant]["atom"] != 0
            or statistics[variant]["edge"] != 0
        ):
            raise AssertionError(
                f"{case}/{variant}: normal padding unexpectedly recompiled: "
                f"{statistics[variant]}"
            )
        if case == "low_padding" and (
            statistics[variant]["atom"] < 1
            or statistics[variant]["edge"] < 1
        ):
            raise AssertionError(
                f"{case}/{variant}: compression did not demonstrate both "
                f"recompilation causes: {statistics[variant]}"
            )
    return {"metrics": metrics, "statistics": statistics}


def run_newton_cases(
    *,
    args: argparse.Namespace,
    script_directory: Path,
    output_directory: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    """Check the supported Newton-off fallback and rejected combination."""
    successful: dict[str, tuple[Path, subprocess.CompletedProcess[str]]] = {}
    for name, communication, newton in (
        ("newton_on_default", "off", "on"),
        ("newton_off_default", "off", "off"),
    ):
        dump = output_directory / f"{name}.lammpstrj"
        command = [
            *shlex.split(args.launcher),
            args.lmp,
            "-var",
            "model",
            str(args.model.resolve()),
            "-var",
            "comm",
            communication,
            "-var",
            "newton_setting",
            newton,
            "-var",
            "prediction_dump",
            str(dump),
            "-log",
            str(output_directory / f"{name}.log"),
            "-in",
            "newton.lmp",
        ]
        successful[name] = (
            dump,
            run_command(
                name,
                command,
                working_directory=script_directory,
                output_directory=output_directory,
                environment=environment,
            ),
        )

    metrics = compare_lammps_runs(
        "newton_off_default_fallback",
        successful["newton_on_default"][0],
        successful["newton_off_default"][0],
        successful["newton_on_default"][1].stdout,
        successful["newton_off_default"][1].stdout,
    )

    rejected_name = "newton_off_comm_rejected"
    rejected = run_command(
        rejected_name,
        [
            *shlex.split(args.launcher),
            args.lmp,
            "-var",
            "model",
            str(args.model.resolve()),
            "-var",
            "comm",
            "on",
            "-var",
            "newton_setting",
            "off",
            "-var",
            "prediction_dump",
            str(output_directory / f"{rejected_name}.lammpstrj"),
            "-log",
            str(output_directory / f"{rejected_name}.log"),
            "-in",
            "newton.lmp",
        ],
        working_directory=script_directory,
        output_directory=output_directory,
        environment=environment,
        expect_success=False,
    )
    expected_error = "Communication requires Newton pair forces"
    if expected_error not in rejected.stdout:
        raise AssertionError(
            f"{rejected_name}: missing documented error {expected_error!r}"
        )
    print("PASS Newton behavior", flush=True)
    return {"fallback_metrics": metrics, "rejected_error": expected_error}


def prepare_molecular_frames(
    samples_path: Path,
    output_directory: Path,
) -> tuple[Path, Path, Path]:
    """Center every molecule on the rank boundary and write rerun inputs."""
    structures = ase.io.read(samples_path, index=":")
    if not structures:
        raise ValueError(f"No molecular frames found in {samples_path}")
    centered_path = output_directory / "molecule_centered.xyz"
    data_path = output_directory / "molecule.lmpdat"
    dump_path = output_directory / "molecule_input.lammpstrj"

    first_numbers = structures[0].numbers.tolist()
    with (
        centered_path.open("w") as centered,
        data_path.open("w") as data,
        dump_path.open("w") as dump,
    ):
        first_positions = None
        for frame_index, structure in enumerate(structures):
            if structure.numbers.tolist() != first_numbers:
                raise ValueError(
                    f"Frame {frame_index} changes atom count, order, or species"
                )
            positions = structure.positions.copy()
            positions[:, 0] -= 0.5 * (
                positions[:, 0].min() + positions[:, 0].max()
            )
            positions[:, 1:] -= positions[:, 1:].mean(axis=0)
            structure.positions = positions
            left = positions[:, 0] < 0.0
            right = positions[:, 0] >= 0.0
            distances = onp.linalg.norm(
                positions[left, None, :] - positions[None, right, :],
                axis=-1,
            )
            if not left.any() or not right.any() or not onp.any(
                distances < MODEL_CUTOFF_ANGSTROM
            ):
                raise AssertionError(
                    f"Frame {frame_index} has no cutoff neighbor across x=0"
                )
            ase.io.write(centered, structure, format="extxyz")
            if first_positions is None:
                first_positions = positions.copy()

            dump.write("ITEM: TIMESTEP\n")
            dump.write(f"{frame_index}\n")
            dump.write("ITEM: NUMBER OF ATOMS\n")
            dump.write(f"{len(structure)}\n")
            dump.write("ITEM: BOX BOUNDS pp pp pp\n")
            dump.write("-25 25\n-25 25\n-25 25\n")
            dump.write("ITEM: ATOMS id x y z\n")
            for atom_id, position in enumerate(positions, start=1):
                dump.write(
                    f"{atom_id} {position[0]:.16g} {position[1]:.16g} "
                    f"{position[2]:.16g}\n"
                )

        data.write("Centered molecular regression frame\n\n")
        data.write(f"{len(first_numbers)} atoms\n")
        data.write("90 atom types\n\n")
        data.write("-25 25 xlo xhi\n-25 25 ylo yhi\n-25 25 zlo zhi\n\n")
        data.write("Atoms\n\n")
        for atom_id, (atomic_number, position) in enumerate(
            zip(first_numbers, first_positions, strict=True),
            start=1,
        ):
            data.write(
                f"{atom_id} {atomic_number} {position[0]:.16g} "
                f"{position[1]:.16g} {position[2]:.16g}\n"
            )
    return centered_path, data_path, dump_path


def run_molecular_prediction_case(
    *,
    args: argparse.Namespace,
    script_directory: Path,
    output_directory: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    """Compare split-rank molecular predictions with each other and MACE."""
    centered, data_file, rerun_dump = prepare_molecular_frames(
        args.samples.resolve(), output_directory
    )
    reference_path = output_directory / "mace_reference.xyz"
    run_command(
        "mace_reference",
        [
            args.mace_eval,
            "--configs",
            str(centered),
            "--model",
            str(args.reference_model),
            "--output",
            str(reference_path),
            "--device",
            args.mace_device,
            "--default_dtype",
            "float64",
            "--batch_size",
            "16",
            "--head",
            "default",
        ],
        working_directory=script_directory,
        output_directory=output_directory,
        environment=environment,
    )

    runs: dict[
        str, tuple[list[dict[str, onp.ndarray]], onp.ndarray]
    ] = {}
    for variant, communication in (("default", "off"), ("comm", "on")):
        name = f"molecule_{variant}"
        prediction = output_directory / f"{name}.lammpstrj"
        completed = run_command(
            name,
            [
                *shlex.split(args.launcher),
                args.lmp,
                "-var",
                "model",
                str(args.model.resolve()),
                "-var",
                "comm",
                communication,
                "-var",
                "data_file",
                str(data_file),
                "-var",
                "rerun_dump",
                str(rerun_dump),
                "-var",
                "prediction_dump",
                str(prediction),
                "-log",
                str(output_directory / f"{name}.log"),
                "-in",
                "predict.lmp",
            ],
            working_directory=script_directory,
            output_directory=output_directory,
            environment=environment,
        )
        frames = read_lammps_dump(prediction)
        for frame_index, frame in enumerate(frames):
            owners = set(frame["proc"].astype(int))
            if owners != {0, 1}:
                raise AssertionError(
                    f"{name}, frame {frame_index}: expected ownership on "
                    f"ranks 0 and 1, got {sorted(owners)}"
                )
        runs[variant] = (frames, read_potential_energies(completed.stdout))

    references = ase.io.read(reference_path, index=":")
    if len(references) != len(runs["default"][0]):
        raise AssertionError("MACE and LAMMPS molecular frame counts differ")

    result: dict[str, object] = {}
    for variant, (frames, energies) in runs.items():
        force_errors = []
        energy_errors = []
        for frame_index, (reference, frame, energy) in enumerate(
            zip(references, frames, energies, strict=True)
        ):
            force_key = next(
                (
                    key
                    for key in ("MACE_forces", "forces")
                    if key in reference.arrays
                ),
                None,
            )
            energy_key = next(
                (
                    key
                    for key in ("MACE_energy", "energy")
                    if key in reference.info
                ),
                None,
            )
            if force_key is None or energy_key is None:
                raise KeyError(
                    f"MACE frame {frame_index} lacks force keys "
                    "MACE_forces/forces or energy keys MACE_energy/energy"
                )
            reference_forces = onp.asarray(reference.arrays[force_key])
            reference_energy = float(reference.info[energy_key])
            onp.testing.assert_array_equal(
                frame["type"].astype(int),
                reference.numbers,
                err_msg=(
                    f"molecule_{variant}, frame {frame_index}: atomic "
                    "numbers differ from the independent MACE input"
                ),
            )
            predicted_forces = onp.column_stack(
                [frame[name] for name in ("fx", "fy", "fz")]
            )
            if predicted_forces.shape != reference_forces.shape:
                raise AssertionError(
                    f"molecule_{variant}, frame {frame_index}: force shapes "
                    f"differ"
                )
            force_errors.append(predicted_forces - reference_forces)
            energy_errors.append(
                abs(float(energy) - reference_energy) / len(reference)
            )
        force_error_array = onp.asarray(force_errors)
        maximum_force_index = onp.unravel_index(
            onp.argmax(onp.abs(force_error_array)), force_error_array.shape
        )
        maximum_force_error = float(abs(force_error_array[maximum_force_index]))
        maximum_energy_frame = int(onp.argmax(energy_errors))
        maximum_energy_error = float(energy_errors[maximum_energy_frame])
        if maximum_force_error > MACE_FORCE_ERROR_EV_PER_ANGSTROM:
            raise AssertionError(
                f"molecule_{variant}: MACE force error "
                f"{maximum_force_error:.6g} eV/Angstrom exceeds "
                f"{MACE_FORCE_ERROR_EV_PER_ANGSTROM:.6g} at frame "
                f"{maximum_force_index[0]}, atom {maximum_force_index[1] + 1}, "
                f"component {maximum_force_index[2]}"
            )
        if maximum_energy_error > MACE_ENERGY_ERROR_EV_PER_ATOM:
            raise AssertionError(
                f"molecule_{variant}: MACE energy error "
                f"{maximum_energy_error:.6g} eV/atom exceeds "
                f"{MACE_ENERGY_ERROR_EV_PER_ATOM:.6g} at frame "
                f"{maximum_energy_frame}"
            )
        result[f"{variant}_versus_mace"] = {
            "maximum_force_error_ev_per_angstrom": maximum_force_error,
            "maximum_energy_error_ev_per_atom": maximum_energy_error,
        }

    default_frames, default_energies = runs["default"]
    communication_frames, communication_energies = runs["comm"]
    if len(default_frames) != len(communication_frames):
        raise AssertionError("Molecular comm/default frame counts differ")
    if default_energies.shape != communication_energies.shape:
        raise AssertionError("Molecular comm/default energy shapes differ")
    direct_force_errors = []
    direct_atomic_energy_errors = []
    for frame_index, (expected, actual) in enumerate(
        zip(default_frames, communication_frames, strict=True)
    ):
        for column in ("id", "type", "step"):
            onp.testing.assert_array_equal(
                actual[column],
                expected[column],
                err_msg=(
                    f"molecule comm/default, frame {frame_index}: "
                    f"{column} differs"
                ),
            )
        direct_force_errors.append(
            onp.column_stack(
                [actual[name] - expected[name] for name in ("fx", "fy", "fz")]
            )
        )
        direct_atomic_energy_errors.append(
            actual["c_atom_energy"] - expected["c_atom_energy"]
        )
    direct_force_error_array = onp.abs(onp.asarray(direct_force_errors))
    direct_force_index = onp.unravel_index(
        onp.argmax(direct_force_error_array), direct_force_error_array.shape
    )
    direct_force_error = float(direct_force_error_array[direct_force_index])
    direct_atomic_energy_error_array = onp.abs(
        onp.asarray(direct_atomic_energy_errors)
    )
    direct_atomic_energy_index = onp.unravel_index(
        onp.argmax(direct_atomic_energy_error_array),
        direct_atomic_energy_error_array.shape,
    )
    direct_atomic_energy_error = float(
        direct_atomic_energy_error_array[direct_atomic_energy_index]
    )
    if direct_force_error > MAX_FORCE_ERROR_EV_PER_ANGSTROM:
        raise AssertionError(
            "molecule communication/default force error "
            f"{direct_force_error:.6g} eV/Angstrom exceeds "
            f"{MAX_FORCE_ERROR_EV_PER_ANGSTROM:.6g} at frame "
            f"{direct_force_index[0]}, atom {direct_force_index[1] + 1}, "
            f"component {direct_force_index[2]}"
        )
    if direct_atomic_energy_error > MAX_ATOMIC_ENERGY_ERROR_EV:
        raise AssertionError(
            "molecule communication/default atomic energy error "
            f"{direct_atomic_energy_error:.6g} eV exceeds "
            f"{MAX_ATOMIC_ENERGY_ERROR_EV:.6g} at frame "
            f"{direct_atomic_energy_index[0]}, atom ID "
            f"{int(default_frames[direct_atomic_energy_index[0]]['id'][direct_atomic_energy_index[1]])}"
        )
    result["comm_versus_default"] = {
        "maximum_force_error_ev_per_angstrom": direct_force_error,
        "maximum_atomic_energy_error_ev": direct_atomic_energy_error,
    }
    print(
        f"PASS molecular predictions: frames={len(references)}, "
        f"comm/default atomic dE={direct_atomic_energy_error:.3e} eV, "
        f"dF={direct_force_error:.3e} eV/Angstrom",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export and test chemtrain's default and distributed MACE variants."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Model bundle path; defaults to OUTPUT_DIRECTORY/model.ptb.",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        help="Molecular XYZ input; defaults to the bundled samples.xyz.",
    )
    parser.add_argument(
        "--reference_model",
        type=Path,
        help=(
            "Torch reference model; defaults to "
            "OUTPUT_DIRECTORY/reference.model."
        ),
    )
    parser.add_argument("--lmp", default="lmp", help="LAMMPS executable.")
    parser.add_argument(
        "--launcher",
        default="mpirun -np 2",
        help="Two-rank MPI launcher; the tests require exactly two ranks.",
    )
    parser.add_argument(
        "--mace_eval", default="mace_eval_configs", help="MACE CLI executable."
    )
    parser.add_argument(
        "--mace_device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used only by the independent MACE CLI reference.",
    )
    parser.add_argument(
        "--output_directory",
        type=Path,
        default=Path("results"),
        help="Directory for the model, logs, dumps, and JSON summary.",
    )
    parser.add_argument(
        "--skip_export",
        action="store_true",
        help="Use the model files supplied by --model and --reference_model.",
    )
    args = parser.parse_args()

    script_directory = Path(__file__).resolve().parent
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    args.model = (
        (output_directory / "model.ptb")
        if args.model is None
        else args.model.resolve()
    )
    args.reference_model = (
        (output_directory / "reference.model")
        if args.reference_model is None
        else args.reference_model.resolve()
    )
    args.samples = (
        (script_directory / "samples.xyz")
        if args.samples is None
        else args.samples.resolve()
    )
    environment = os.environ.copy()
    environment["JCN_VALIDATE_COMMUNICATION"] = "1"
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")

    if not args.skip_export:
        run_command(
            "export",
            [
                sys.executable,
                "export_model.py",
                "--output",
                str(args.model),
                "--reference_output",
                str(args.reference_model),
            ],
            working_directory=script_directory,
            output_directory=output_directory,
            environment=environment,
        )
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if not args.samples.exists():
        raise FileNotFoundError(args.samples)
    if not args.reference_model.exists():
        raise FileNotFoundError(args.reference_model)

    summary = {
        "normal_padding": run_trajectory_case(
            "normal_padding",
            2.0,
            4.0,
            args=args,
            script_directory=script_directory,
            output_directory=output_directory,
            environment=environment,
        ),
        "low_padding": run_trajectory_case(
            "low_padding",
            1.01,
            1.01,
            args=args,
            script_directory=script_directory,
            output_directory=output_directory,
            environment=environment,
        ),
        "newton": run_newton_cases(
            args=args,
            script_directory=script_directory,
            output_directory=output_directory,
            environment=environment,
        ),
        "molecule": run_molecular_prediction_case(
            args=args,
            script_directory=script_directory,
            output_directory=output_directory,
            environment=environment,
        ),
    }
    summary_path = output_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nPASS communication regression; summary: {summary_path}")


if __name__ == "__main__":
    main()
