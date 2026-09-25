#pragma once

#include <util/types.hpp>
#include "Emu/RSX/RSXThread.h"
#include "Emu/RSX/Host/MM.h"

#include "context_accessors.define.h"

namespace rsx
{
	namespace util
	{
		template <bool FlushDMA, bool FlushPipe>
		static void write_gcm_label(context* ctx, u32 type, u32 address, u32 data)
		{
			// The RSX uses the unprotected mapping for labels. The label page can hold zcull reports too (16 KiB host
			// pages: labels and report slots 0-703 share one page), whose pages are access-protected to detect CPU reads
			// of pending reports. The RSX's own label traffic must not count as such a read.
			auto& label = *vm::get_super_ptr<atomic_t<RsxSemaphore>>(address);

			const bool is_flip_sema = (address == (RSX(ctx)->label_addr + 0x10) || address == (RSX(ctx)->device_addr + 0x30));
			if (!is_flip_sema)
			{
				// First, queue the GPU work. If it flushes the queue for us, the following routines will be faster.
				const bool handled = RSX(ctx)->get_backend_config().supports_host_gpu_labels && RSX(ctx)->release_GCM_label(type, address, data);

				// (A deferred write to the same address would change the value later, see flush_deferred_labels)
				if (label.load() == data && !RSX(ctx)->has_deferred_label_at(address))
				{
					// It's a no-op to write the same value (although there is a delay in real-hw so it's more accurate to allow GPU label in this case)
					// There is no possible way for the guest to know that the label has been processed so we can skip MM sync here.
					return;
				}

				if constexpr (FlushDMA || FlushPipe)
				{
					if constexpr (FlushDMA)
					{
						// Release op must be acoompanied by MM flush.
						// FlushPipe implicitly does a MM flush but FlushDMA does not. Trigger the flush here
						rsx::mm_flush();

						// If the backend handled the request, this call will basically be a NOP
						g_fxo->get<rsx::dma_manager>().sync();
					}

					if constexpr (FlushPipe)
					{
						// Syncronization point, may be associated with memory changes without actually changing addresses
						RSX(ctx)->m_graphics_state |= rsx::pipeline_state::fragment_program_needs_rehash;

						// Manually flush the pipeline.
						// It is possible to stream report writes using the host GPU, but that generates too much submit traffic.
						RSX(ctx)->sync();
					}
				}

				if (handled)
				{
					// Backend will handle it, nothing to write.
					return;
				}
			}

			// Labels are strongly ordered: texture read labels held back for zcull reports go first
			RSX(ctx)->flush_deferred_labels();

			label.store(data);
		}
	}
}

#include "context_accessors.undef.h"
