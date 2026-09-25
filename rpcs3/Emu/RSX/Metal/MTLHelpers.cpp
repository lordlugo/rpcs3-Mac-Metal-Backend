#include "stdafx.h"
#include "MTLHelpers.h"
#include "mtlutils/device.h"
#include "Emu/RSX/Common/bitfield.hpp"
#include "Emu/system_config.h"

namespace mtl
{
	static rsx::atomic_bitmask_t<runtime_state, u64> g_runtime_state;
	static u64 g_num_processed_frames = 0;
	static u64 g_num_total_frames = 0;

	const mtl::render_device* get_current_renderer()
	{
		return g_render_device;
	}

	bool emulate_primitive_restart(rsx::primitive_type type)
	{
		// Metal always restarts strips on 0xFFFF / 0xFFFFFFFF. Lists never restart.
		// RSX restart index is rewritten by the common index upload code; only non-strip topologies need emulation.
		switch (type)
		{
		case rsx::primitive_type::triangle_strip:
		case rsx::primitive_type::line_strip:
			return false;
		default:
			return true;
		}
	}

	bool sanitize_fp_values()
	{
		return false;
	}

	bool emulate_conditional_rendering()
	{
		// Metal has no predication: always use the shader predicate path.
		return true;
	}

	void raise_status_interrupt(runtime_state status)
	{
		g_runtime_state |= status;
	}

	void clear_status_interrupt(runtime_state status)
	{
		g_runtime_state.clear(status);
	}

	bool test_status_interrupt(runtime_state status)
	{
		return g_runtime_state & status;
	}

	void enter_uninterruptible()
	{
		raise_status_interrupt(runtime_state::uninterruptible);
	}

	void leave_uninterruptible()
	{
		clear_status_interrupt(runtime_state::uninterruptible);
	}

	bool is_uninterruptible()
	{
		return test_status_interrupt(runtime_state::uninterruptible);
	}

	void advance_completed_frame_counter()
	{
		g_num_processed_frames++;
	}

	void advance_frame_counter()
	{
		ensure(g_num_processed_frames <= g_num_total_frames);
		g_num_total_frames++;
	}

	u64 get_current_frame_id()
	{
		return g_num_total_frames;
	}

	u64 get_last_completed_frame_id()
	{
		return (g_num_processed_frames > 0) ? g_num_processed_frames - 1 : 0;
	}

	void reset_runtime_state()
	{
		g_runtime_state.clear();
		g_num_processed_frames = 0;
		g_num_total_frames = 0;
	}
}
