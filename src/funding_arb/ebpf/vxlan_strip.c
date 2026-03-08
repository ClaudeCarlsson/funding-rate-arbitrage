#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

#define VXLAN_PORT 4789

struct vxlanhdr {
    __u32 vx_flags;
    __u32 vx_vni;
};

SEC("xdp")
int xdp_vxlan_decapsulate(struct xdp_md *ctx) {
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;

    if (eth->h_proto != bpf_htons(ETH_P_IP))
        return XDP_PASS;

    struct iphdr *iph = (void *)(eth + 1);
    if ((void *)(iph + 1) > data_end)
        return XDP_PASS;

    if (iph->protocol != IPPROTO_UDP)
        return XDP_PASS;

    struct udphdr *udph = (void *)(iph + 1);
    if ((void *)(udph + 1) > data_end)
        return XDP_PASS;

    // Check if it's a VxLAN packet on the standard port
    if (udph->dest != bpf_htons(VXLAN_PORT))
        return XDP_PASS;

    struct vxlanhdr *vxh = (void *)(udph + 1);
    if ((void *)(vxh + 1) > data_end)
        return XDP_PASS;

    // Calculate the total encapsulation offset to strip
    // Outer ETH (14) + Outer IP (20) + Outer UDP (8) + VxLAN (8) = 50 bytes
    unsigned int off = sizeof(struct ethhdr) + sizeof(struct iphdr) + 
                       sizeof(struct udphdr) + sizeof(struct vxlanhdr);

    // We keep the original outer Ethernet header but move it forward to the inner payload
    // To properly decapsulate, we should pop the headers and present the inner frame.
    // However, for pure Aeron UDP payloads, we can just adjust the data pointer.
    if (bpf_xdp_adjust_head(ctx, off))
        return XDP_DROP;

    // Return XDP_PASS to let the stripped packet continue to the shared-memory buffer
    return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
