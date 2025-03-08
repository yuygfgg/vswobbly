"""
WobblyParser - A VapourSynth parser for Wobbly project files (.wob)

This module provides functionality to directly load and process Wobbly projects
in VapourSynth scripts without needing to export .vpy files.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional, Union, Callable, TypedDict

import vapoursynth as vs
from vapoursynth import core


class FrameProperties(TypedDict, total=False):
    """Type definition for frame properties dictionary"""
    WobblyProject: str
    WobblyVersion: str
    WobblySourceFilter: str
    WobblyCustomList: str
    WobblyCustomListPreset: str
    WobblyCustomListPosition: str
    WobblySectionStart: int
    WobblySectionEnd: int
    WobblySectionPresets: str
    WobblyMatch: str
    WobblyCombed: bool
    WobblyInterlacedFade: bool
    WobblyFieldDifference: float
    WobblyOrphan: bool
    WobblyOrphanType: str
    WobblyOrphanDecimated: bool
    WobblyFrozenFrame: bool
    WobblyFrozenSource: int
    WobblyDecimated: bool
    WobblyCropEarly: bool
    WobblyCropLeft: int
    WobblyCropTop: int
    WobblyCropRight: int
    WobblyCropBottom: int
    WobblyResizeEnabled: bool
    WobblyResizeWidth: int
    WobblyResizeHeight: int
    WobblyResizeFilter: str
    WobblyDepthEnabled: bool
    WobblyDepthBits: int
    WobblyDepthFloat: bool
    WobblyDepthDither: str
    WobblyTrimStart: int
    WobblyTrimEnd: int


class WobblyKeys:
    """Constant key definitions for Wobbly project JSON structure"""
    wobbly_version = "wobbly version"
    project_format_version = "project format version"
    input_file = "input file"
    input_frame_rate = "input frame rate"
    input_resolution = "input resolution"
    trim = "trim"
    source_filter = "source filter"
    user_interface = "user interface"
    vfm_parameters = "vfm parameters"
    matches = "matches"
    original_matches = "original matches"
    sections = "sections"
    presets = "presets"
    frozen_frames = "frozen frames"
    combed_frames = "combed frames"
    interlaced_fades = "interlaced fades"
    decimated_frames = "decimated frames"
    custom_lists = "custom lists"
    resize = "resize"
    crop = "crop"
    depth = "depth"

    class VFMParameters:
        order = "order"

    class Sections:
        start = "start"
        presets = "presets"

    class Presets:
        name = "name"
        contents = "contents"

    class CustomLists:
        name = "name"
        preset = "preset"
        position = "position"
        frames = "frames"

    class Resize:
        width = "width"
        height = "height"
        filter = "filter"
        enabled = "enabled"

    class Crop:
        early = "early"
        left = "left"
        top = "top"
        right = "right"
        bottom = "bottom"
        enabled = "enabled"

    class Depth:
        bits = "bits"
        float_samples = "float samples"
        dither = "dither"
        enabled = "enabled"

    class InterlacedFades:
        frame = "frame"
        field_difference = "field difference"


def wobbly_source(
    wob_project_path: Union[str, Path],
    timecode_output_path: Optional[Union[str, Path]] = None
) -> vs.VideoNode:
    """
    Load and process a video according to a Wobbly project file

    Args:
        wob_project_path: Path to the Wobbly project file (.wob)
        timecode_output_path: Optional path to save generated timecodes (v2 format)

    Returns:
        vs.VideoNode: The processed video according to the Wobbly project
    """
    Keys = WobblyKeys  # Alias for shorter reference

    # Read and parse the Wobbly project file
    try:
        with open(wob_project_path, 'r', encoding='utf-8') as f:
            project = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to read or parse Wobbly project file: {e}")

    # Get input file path
    input_file = project.get(Keys.input_file)
    source_filter = project.get(Keys.source_filter, "")

    if not input_file:
        raise ValueError("No input file specified in the project")

    # Handle relative paths
    if not os.path.isabs(input_file):
        wob_dir = os.path.dirname(os.path.abspath(str(wob_project_path)))
        input_file = os.path.join(wob_dir, input_file)

    if not os.path.exists(input_file):
        raise ValueError(f"Input file does not exist: {input_file}")

    # Create global frame property map - to track properties throughout processing
    # Key: frame number, Value: property dictionary
    frame_props: Dict[int, FrameProperties] = {}

    # Load source video
    try:
        if source_filter == "bs.VideoSource":
            # BestSource loading
            src = core.bs.VideoSource(input_file, rff=True, showprogress=False)
        else:
            # Use specified filter
            filter_parts = source_filter.split('.')
            plugin = getattr(core, filter_parts[0])
            src = getattr(plugin, filter_parts[1])(input_file)
    except Exception as e:
        raise ValueError(f"Failed to load video: {e}")

    # Initialize basic properties for each frame
    for n in range(src.num_frames):
        frame_props[n] = {
            "WobblyProject": os.path.basename(str(wob_project_path)),
            "WobblyVersion": project.get(Keys.wobbly_version, ""),
            "WobblySourceFilter": source_filter,
            # Initialize empty values
            "WobblyCustomList": "",
            "WobblyCustomListPreset": "",
            "WobblyCustomListPosition": "",
            "WobblySectionStart": -1,
            "WobblySectionEnd": -1,
            "WobblySectionPresets": "",
            "WobblyMatch": ""
        }

    # Create preset functions
    presets: Dict[str, Callable[[vs.VideoNode], vs.VideoNode]] = {}
    for preset_info in project.get(Keys.presets, []):
        preset_name = preset_info.get(Keys.Presets.name)
        preset_contents = preset_info.get(Keys.Presets.contents)

        if not preset_name or preset_contents is None:
            continue

        try:
            # Create an executable preset function
            exec_globals = {'vs': vs, 'core': core, 'c': core}
            exec(f"def preset_{preset_name}(clip):\n" +
                 "\n".join("    " + line for line in preset_contents.split('\n')) +
                 "\n    return clip", exec_globals)

            presets[preset_name] = exec_globals[f"preset_{preset_name}"]
        except Exception as e:
            print(f"Warning: Error creating preset '{preset_name}': {e}")

    # Prepare data for processing
    frame_mapping: Dict[int, int] = {}  # Map processed frames to original frames
    for i in range(src.num_frames):
        frame_mapping[i] = i  # Initially the mapping is identical

    # Process video pipeline
    try:
        # Apply early crop
        crop_info = project.get(Keys.crop, {})
        if crop_info.get(Keys.Crop.enabled, False) and crop_info.get(Keys.Crop.early, False):
            crop_props = {
                "WobblyCropEarly": True,
                "WobblyCropLeft": crop_info.get(Keys.Crop.left, 0),
                "WobblyCropTop": crop_info.get(Keys.Crop.top, 0),
                "WobblyCropRight": crop_info.get(Keys.Crop.right, 0),
                "WobblyCropBottom": crop_info.get(Keys.Crop.bottom, 0)
            }

            # Update all frame properties
            for n in frame_props:
                frame_props[n].update(crop_props)

            src = core.std.CropRel(
                clip=src,
                left=crop_info.get(Keys.Crop.left, 0),
                top=crop_info.get(Keys.Crop.top, 0),
                right=crop_info.get(Keys.Crop.right, 0),
                bottom=crop_info.get(Keys.Crop.bottom, 0)
            )

        # Apply trimming
        trim_list = project.get(Keys.trim, [])
        if trim_list:
            clips = []
            new_frame_props: Dict[int, FrameProperties] = {}  # Store new frame property map
            new_frame_idx = 0

            for trim in trim_list:
                first, last = trim
                if first <= last and first < src.num_frames and last < src.num_frames:
                    # Create segment
                    segment = src[first:last+1]

                    # Update frame properties and mapping
                    for i in range(first, last+1):
                        # Update trim info
                        if i in frame_props:
                            props = frame_props[i].copy()
                            props.update({
                                "WobblyTrimStart": first,
                                "WobblyTrimEnd": last
                            })
                            new_frame_props[new_frame_idx] = props
                            # Update mapping
                            frame_mapping[new_frame_idx] = i
                            new_frame_idx += 1

                    clips.append(segment)

            if clips:
                src = core.std.Splice(clips=clips)
                frame_props = new_frame_props  # Update frame property map

        # Apply custom lists - PostSource
        src, frame_props, frame_mapping = apply_custom_lists(
            src, project, presets, "post source", Keys, frame_props, frame_mapping
        )

        # Get match data
        matches_list = project.get(Keys.matches)
        original_matches_list = project.get(Keys.original_matches)

        # Ensure we have match data for later
        matches = ""
        if matches_list:
            matches = "".join(matches_list)
        elif original_matches_list:
            matches = "".join(original_matches_list)

        # Record match for each frame
        if matches:
            for n in frame_props:
                orig_frame = frame_mapping[n]
                if orig_frame < len(matches):
                    frame_props[n]["WobblyMatch"] = matches[orig_frame]

        # Apply FieldHint
        if hasattr(core, 'fh'):
            if matches_list:
                vfm_params = project.get(Keys.vfm_parameters, {})
                order = vfm_params.get(Keys.VFMParameters.order, 1)
                src = core.fh.FieldHint(clip=src, tff=order, matches=matches)
            elif original_matches_list:
                vfm_params = project.get(Keys.vfm_parameters, {})
                order = vfm_params.get(Keys.VFMParameters.order, 1)
                src = core.fh.FieldHint(clip=src, tff=order, matches=matches)

        # Apply custom lists - PostFieldMatch
        src, frame_props, frame_mapping = apply_custom_lists(
            src, project, presets, "post field match", Keys, frame_props, frame_mapping
        )

        # Apply sections and record section info
        sections_list = project.get(Keys.sections, [])

        if sections_list:
            # Sort sections by start frame
            sorted_sections = sorted(sections_list, key=lambda s: s.get(Keys.Sections.start, 0))

            # Mark each frame with its section
            for i, section_info in enumerate(sorted_sections):
                start = section_info.get(Keys.Sections.start, 0)
                next_start = sorted_sections[i+1].get(Keys.Sections.start, src.num_frames) if i+1 < len(sorted_sections) else src.num_frames

                section_presets = section_info.get(Keys.Sections.presets, [])
                presets_str = ",".join(section_presets)

                # Update frame properties
                for n in frame_props:
                    orig_frame = frame_mapping[n]
                    if start <= orig_frame < next_start:
                        frame_props[n].update({
                            "WobblySectionStart": start,
                            "WobblySectionEnd": next_start-1,
                            "WobblySectionPresets": presets_str
                        })

            # Apply presets and splice
            sections = []
            new_frame_props: Dict[int, FrameProperties] = {}
            new_frame_idx = 0

            for i, section_info in enumerate(sorted_sections):
                start = section_info.get(Keys.Sections.start, 0)
                next_start = sorted_sections[i+1].get(Keys.Sections.start, src.num_frames) if i+1 < len(sorted_sections) else src.num_frames

                # Apply presets
                section_clip = src[start:next_start]
                for preset_name in section_info.get(Keys.Sections.presets, []):
                    if preset_name in presets:
                        section_clip = presets[preset_name](section_clip)

                # Update frame mapping and properties
                for j in range(section_clip.num_frames):
                    src_idx = start + j
                    if src_idx < len(frame_mapping):
                        orig_frame = frame_mapping[src_idx]
                        # Copy original frame properties
                        if src_idx in frame_props:
                            new_frame_props[new_frame_idx] = frame_props[src_idx].copy()
                            # Update mapping
                            frame_mapping[new_frame_idx] = orig_frame
                            new_frame_idx += 1

                sections.append(section_clip)

            # Merge all sections
            if sections:
                src = core.std.Splice(clips=sections, mismatch=True)
                frame_props = new_frame_props  # Update frame properties

        # Prepare special frame information
        combed_frames = set(project.get(Keys.combed_frames, []))
        decimated_frames = set(project.get(Keys.decimated_frames, []))

        # Process Interlaced Fades
        interlaced_fades = project.get(Keys.interlaced_fades, [])
        fade_dict: Dict[int, float] = {}

        if interlaced_fades:
            for fade in interlaced_fades:
                frame = fade.get(Keys.InterlacedFades.frame)
                field_diff = fade.get(Keys.InterlacedFades.field_difference, 0)
                if frame is not None:
                    fade_dict[frame] = field_diff

        # Identify and mark Orphan Fields
        orphan_fields: Dict[int, Dict[str, Any]] = {}

        if matches and sections_list:
            # Sort sections by start frame
            sorted_sections = sorted(sections_list, key=lambda s: s.get(Keys.Sections.start, 0))
            section_boundaries = [s.get(Keys.Sections.start, 0) for s in sorted_sections]
            section_boundaries.append(src.num_frames)  # Add last frame as boundary

            # Identify orphan fields
            for i in range(len(section_boundaries) - 1):
                section_start = section_boundaries[i]
                section_end = section_boundaries[i+1] - 1

                # Check if section start has 'n' match
                if section_start < len(matches) and matches[section_start] == 'n':
                    orphan_fields[section_start] = {'type': 'n', 'decimated': section_start in decimated_frames}

                # Check if section end has 'b' match
                if section_end < len(matches) and matches[section_end] == 'b':
                    orphan_fields[section_end] = {'type': 'b', 'decimated': section_end in decimated_frames}

        # Update special frame properties
        for n in frame_props:
            orig_frame = frame_mapping[n]
            props = frame_props[n]

            # Mark combed frames
            if orig_frame in combed_frames:
                props["WobblyCombed"] = True

            # Mark interlaced fades
            if orig_frame in fade_dict:
                props["WobblyInterlacedFade"] = True
                props["WobblyFieldDifference"] = fade_dict[orig_frame]

            # Mark orphan fields
            if orig_frame in orphan_fields:
                info = orphan_fields[orig_frame]
                props["WobblyOrphan"] = True
                props["WobblyOrphanType"] = info['type']
                props["WobblyOrphanDecimated"] = info['decimated']

            # Mark decimated frames
            if orig_frame in decimated_frames:
                props["WobblyDecimated"] = True

        # Apply frozen frames
        frozen_frames_list = project.get(Keys.frozen_frames, [])
        if frozen_frames_list and hasattr(core.std, 'FreezeFrames'):
            first_frames = []
            last_frames = []
            replacement_frames = []

            for ff_info in frozen_frames_list:
                if len(ff_info) == 3:
                    first, last, replacement = ff_info
                    if 0 <= first <= last < src.num_frames and 0 <= replacement < src.num_frames:
                        first_frames.append(first)
                        last_frames.append(last)
                        replacement_frames.append(replacement)

                        # Record frozen frame info
                        for i in range(first, last+1):
                            if i in frame_props:
                                frame_props[i]["WobblyFrozenFrame"] = True
                                frame_props[i]["WobblyFrozenSource"] = replacement

            if first_frames:
                src = core.std.FreezeFrames(
                    clip=src,
                    first=first_frames,
                    last=last_frames,
                    replacement=replacement_frames
                )

        # Apply frame rate conversion (delete frames)
        decimated_frames_list = project.get(Keys.decimated_frames, [])
        if decimated_frames_list:
            # Filter valid frames
            frames_to_delete = [f for f in decimated_frames_list if 0 <= f < src.num_frames]

            if frames_to_delete:
                # Create new frame property map
                new_frame_props: Dict[int, FrameProperties] = {}
                new_idx = 0

                for n in range(src.num_frames):
                    orig_frame = frame_mapping.get(n, n)

                    # If not a frame to delete
                    if orig_frame not in frames_to_delete:
                        if n in frame_props:
                            new_frame_props[new_idx] = frame_props[n].copy()
                            new_idx += 1

                # Delete frames
                src = core.std.DeleteFrames(clip=src, frames=frames_to_delete)
                frame_props = new_frame_props  # Update frame properties

        # Apply custom lists - PostDecimate
        src, frame_props, frame_mapping = apply_custom_lists(
            src, project, presets, "post decimate", Keys, frame_props, frame_mapping
        )

        # Apply final crop
        if crop_info.get(Keys.Crop.enabled, False) and not crop_info.get(Keys.Crop.early, False):
            crop_props = {
                "WobblyCropEarly": False,
                "WobblyCropLeft": crop_info.get(Keys.Crop.left, 0),
                "WobblyCropTop": crop_info.get(Keys.Crop.top, 0),
                "WobblyCropRight": crop_info.get(Keys.Crop.right, 0),
                "WobblyCropBottom": crop_info.get(Keys.Crop.bottom, 0)
            }

            # Update all frame properties
            for n in frame_props:
                frame_props[n].update(crop_props)

            src = core.std.CropRel(
                clip=src,
                left=crop_info.get(Keys.Crop.left, 0),
                top=crop_info.get(Keys.Crop.top, 0),
                right=crop_info.get(Keys.Crop.right, 0),
                bottom=crop_info.get(Keys.Crop.bottom, 0)
            )

        # Apply resize and bit depth
        resize_info = project.get(Keys.resize, {})
        depth_info = project.get(Keys.depth, {})

        resize_enabled = resize_info.get(Keys.Resize.enabled, False)
        depth_enabled = depth_info.get(Keys.Depth.enabled, False)

        if resize_enabled or depth_enabled:
            # Record resize and bit depth info
            resize_props: Dict[str, Any] = {}

            resize_filter_name = resize_info.get(Keys.Resize.filter, "Bicubic")
            if resize_filter_name:
                resize_filter_name = resize_filter_name[0].upper() + resize_filter_name[1:]
            else:
                resize_filter_name = "Bicubic"

            if not hasattr(core.resize, resize_filter_name):
                resize_filter_name = "Bicubic"

            resize_args: Dict[str, Any] = {}
            if resize_enabled:
                resize_width = resize_info.get(Keys.Resize.width, src.width)
                resize_height = resize_info.get(Keys.Resize.height, src.height)
                resize_args["width"] = resize_width
                resize_args["height"] = resize_height

                resize_props.update({
                    "WobblyResizeEnabled": True,
                    "WobblyResizeWidth": resize_width,
                    "WobblyResizeHeight": resize_height,
                    "WobblyResizeFilter": resize_filter_name
                })

            if depth_enabled:
                bits = depth_info.get(Keys.Depth.bits, 8)
                float_samples = depth_info.get(Keys.Depth.float_samples, False)
                dither = depth_info.get(Keys.Depth.dither, "")
                sample_type = vs.FLOAT if float_samples else vs.INTEGER

                format_id = core.query_video_format(
                    src.format.color_family,
                    sample_type,
                    bits,
                    src.format.subsampling_w,
                    src.format.subsampling_h
                ).id

                resize_args["format"] = format_id

                resize_props.update({
                    "WobblyDepthEnabled": True,
                    "WobblyDepthBits": bits,
                    "WobblyDepthFloat": float_samples,
                    "WobblyDepthDither": dither
                })

            # Update all frame properties
            for n in frame_props:
                frame_props[n].update(resize_props)

            resize_filter = getattr(core.resize, resize_filter_name)
            src = resize_filter(clip=src, **resize_args)

        # Generate timecodes if requested
        if timecode_output_path:
            timecodes = generate_timecodes_v2(project)
            with open(str(timecode_output_path), 'w', encoding='utf-8') as f:
                f.write(timecodes)

        # Finally: Apply all frame properties
        def apply_frame_props(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
            if n in frame_props:
                fout = f.copy()

                # Add all properties
                for key, value in frame_props[n].items():
                    if value is not None:  # Skip None values
                        # VapourSynth doesn't accept None as frame property value
                        fout.props[key] = value

                return fout
            return f

        # Apply all saved frame properties
        src = core.std.ModifyFrame(src, src, apply_frame_props)

    except Exception as e:
        raise RuntimeError(f"Error processing Wobbly project: {e}")

    return src


def apply_custom_lists(
    src: vs.VideoNode,
    project: Dict[str, Any],
    presets: Dict[str, Callable[[vs.VideoNode], vs.VideoNode]],
    position: str,
    Keys: Any,
    frame_props: Dict[int, FrameProperties],
    frame_mapping: Dict[int, int]
) -> Tuple[vs.VideoNode, Dict[int, FrameProperties], Dict[int, int]]:
    """
    Apply custom lists at the specified position

    Args:
        src: Current video source
        project: Project data
        presets: Preset functions
        position: Custom list position
        Keys: Key name constants
        frame_props: Global frame property map
        frame_mapping: Map of processed frames to original frames

    Returns:
        Tuple of (processed_clip, updated_frame_props, updated_frame_mapping)
    """

    custom_lists = [cl for cl in project.get(Keys.custom_lists, []) if cl.get(Keys.CustomLists.position) == position]

    if not custom_lists:
        return src, frame_props, frame_mapping

    all_ranges: List[Tuple[int, int, str, str]] = []  # all covered ranges

    for cl_info in custom_lists:
        cl_name = cl_info.get(Keys.CustomLists.name)
        cl_preset = cl_info.get(Keys.CustomLists.preset)
        cl_frames = cl_info.get(Keys.CustomLists.frames, [])

        # Check if we have preset and frame ranges
        if not cl_preset or not cl_frames:
            continue

        # Check if preset exists
        if cl_preset not in presets:
            continue

        try:
            # Ensure cl_frames is a list of lists
            ranges: List[Tuple[int, int]] = []
            for frame_range in cl_frames:
                if isinstance(frame_range, list) and len(frame_range) == 2:
                    start, end = frame_range

                    # Record all qualifying frames, and update frame properties
                    for frame in range(start, end+1):
                        for n in frame_props:
                            if frame_mapping[n] == frame:
                                frame_props[n].update({
                                    "WobblyCustomList": cl_name,
                                    "WobblyCustomListPreset": cl_preset,
                                    "WobblyCustomListPosition": position
                                })

                    ranges.append((start, end))
                    all_ranges.append((start, end, cl_name, cl_preset))

            # Sort the ranges
            ranges.sort()

            # Apply preset to clip
            if ranges:
                # Create marked segments
                marked_clips = []
                last_end = 0

                for range_start, range_end in ranges:
                    # Ensure valid range
                    if not (0 <= range_start <= range_end < src.num_frames):
                        continue

                    if range_start > last_end:
                        marked_clips.append(src[last_end:range_start])

                    # Apply preset to current range
                    list_clip = presets[cl_preset](src[range_start:range_end+1])
                    marked_clips.append(list_clip)

                    last_end = range_end + 1

                if last_end < src.num_frames:
                    marked_clips.append(src[last_end:])

                if marked_clips:
                    src = core.std.Splice(clips=marked_clips, mismatch=True)
        except Exception as e:
            print(f"Warning: Error applying custom list '{cl_name}': {e}")

    return src, frame_props, frame_mapping


def get_decimation_info(project: Dict[str, Any]) -> Tuple[Dict[int, Set[int]], List[Dict[str, int]]]:
    """
    Get decimation cycle information from the project

    Args:
        project: The Wobbly project data

    Returns:
        Tuple of (decimated_by_cycle, decimation_ranges)
    """
    # Get decimated frames and project length
    decimated_frames: List[int] = project.get('decimated frames', [])

    # Calculate total frames from trim data
    num_frames = 0
    if 'trim' in project:
        for trim in project['trim']:
            if isinstance(trim, list) and len(trim) >= 2:
                num_frames += trim[1] - trim[0] + 1

    # Group decimated frames by cycle
    decimated_by_cycle: Dict[int, Set[int]] = {}
    for frame in decimated_frames:
        cycle = frame // 5
        if cycle not in decimated_by_cycle:
            decimated_by_cycle[cycle] = set()
        decimated_by_cycle[cycle].add(frame % 5)

    # Calculate decimation ranges
    ranges: List[Dict[str, int]] = []
    current_count = -1
    current_start = 0

    for cycle in range((num_frames + 4) // 5):
        count = len(decimated_by_cycle.get(cycle, set()))
        if count != current_count:
            if current_count != -1:
                ranges.append({
                    'start': current_start,
                    'end': cycle * 5,
                    'dropped': current_count
                })
            current_count = count
            current_start = cycle * 5

    if current_count != -1:
        ranges.append({
            'start': current_start,
            'end': num_frames,
            'dropped': current_count
        })

    return decimated_by_cycle, ranges


def frame_number_after_decimation(frame: int, decimated_by_cycle: Dict[int, Set[int]]) -> int:
    """
    Calculate frame number after decimation

    Args:
        frame: Original frame number
        decimated_by_cycle: Dictionary mapping cycles to sets of decimated offsets

    Returns:
        Frame number after decimation
    """
    if frame < 0:
        return 0

    cycle = frame // 5
    offset = frame % 5

    # Count decimated frames before this one
    decimated_before = 0
    for c in range(cycle):
        decimated_before += len(decimated_by_cycle.get(c, set()))

    for o in range(offset):
        if o in decimated_by_cycle.get(cycle, set()):
            decimated_before += 1

    return frame - decimated_before


def generate_timecodes_v1(project: Dict[str, Any]) -> str:
    """
    Generate timecodes in v1 format

    Args:
        project: The Wobbly project data

    Returns:
        String containing timecodes in v1 format
    """
    DEFAULT_FPS = 24000 / 1001

    decimated_by_cycle, ranges = get_decimation_info(project)

    tc = "# timecode format v1\n"
    tc += f"Assume {DEFAULT_FPS:.12f}\n"

    numerators = [30000, 24000, 18000, 12000, 6000]
    denominator = 1001

    for range_info in ranges:
        dropped = range_info['dropped']

        if numerators[dropped] != 24000:
            start_frame = frame_number_after_decimation(range_info['start'], decimated_by_cycle)
            end_frame = frame_number_after_decimation(range_info['end'] - 1, decimated_by_cycle)

            fps = numerators[dropped] / denominator
            tc += f"{start_frame},{end_frame},{fps:.12f}\n"

    return tc


def generate_timecodes_v2(project: Dict[str, Any]) -> str:
    """
    Generate timecodes in v2 format

    Args:
        project: The Wobbly project data

    Returns:
        String containing timecodes in v2 format
    """
    decimated_by_cycle, ranges = get_decimation_info(project)

    tc = "# timecode format v2\n"

    numerators = [30000, 24000, 18000, 12000, 6000]
    denominator = 1001

    # Calculate total output frames
    total_frames = 0
    for range_info in ranges:
        start = range_info['start']
        end = range_info['end']
        total_frames += frame_number_after_decimation(end - 1, decimated_by_cycle) - frame_number_after_decimation(start, decimated_by_cycle) + 1

    current_frame = 0
    current_time_ms = 0.0

    for range_info in ranges:
        dropped = range_info['dropped']
        fps = numerators[dropped] / denominator
        frame_duration_ms = 1000.0 / fps

        start_frame = frame_number_after_decimation(range_info['start'], decimated_by_cycle)
        end_frame = frame_number_after_decimation(range_info['end'] - 1, decimated_by_cycle)

        for _ in range(start_frame, end_frame + 1):
            tc += f"{current_time_ms:.6f}\n"
            current_time_ms += frame_duration_ms
            current_frame += 1

    return tc
