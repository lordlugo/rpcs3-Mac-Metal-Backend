#include "stdafx.h"
#include "MTLDeviceQuery.h"
#include "MTLGSRender.h"
#include "Emu/IdManager.h"

namespace mtl
{
	void create_render_thread(utils::serial* ar)
	{
		g_fxo->init<rsx::thread, named_thread<MTLGSRender>>(ar);
	}
}
