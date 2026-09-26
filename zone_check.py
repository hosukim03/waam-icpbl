import numpy as np
B = {1: np.array([-1400.0, 0.0]), 3: np.array([700.0, 1212.435565])}
DIR = {1: 180.0, 3: 60.0}
REACH, WS = 1500.0, 700.0

def zone_pts(rid, r0, gap_deg, dr=10, da=0.5):
    half = 60.0 - gap_deg / 2
    rs = np.arange(r0, WS + 1e-9, dr)
    angs = np.deg2rad(np.arange(DIR[rid] - half, DIR[rid] + half + 1e-9, da))
    R, A = np.meshgrid(rs, angs)
    P = np.c_[(R * np.cos(A)).ravel(), (R * np.sin(A)).ravel()]
    ok = np.linalg.norm(P - B[rid], axis=1) <= REACH
    return P[ok]

def seg_dist(a0, a1, b0, b1):
    # 2D segment-segment min distance, vectorised; a0,b0 are fixed bases
    def pt_seg(p, s0, s1):
        d = s1 - s0; t = np.clip(((p - s0) * d).sum(-1) / (d * d).sum(-1), 0, 1)
        return np.linalg.norm(p - (s0 + d * t[..., None]), axis=-1)
    def cross(o, a, b): return (a[..., 0]-o[..., 0])*(b[..., 1]-o[..., 1]) - (a[..., 1]-o[..., 1])*(b[..., 0]-o[..., 0])
    inter = (cross(b0, b1, a0) * cross(b0, b1, a1) < 0) & (cross(a0, a1, b0) * cross(a0, a1, b1) < 0)
    d = np.minimum.reduce([pt_seg(a0, b0, b1), pt_seg(a1, b0, b1), pt_seg(b0, a0, a1), pt_seg(b1, a0, a1)])
    return np.where(inter, 0.0, d)

def worst(r0, gap):
    P1, P3 = zone_pts(1, r0, gap), zone_pts(3, r0, gap)
    # restrict to points near the shared boundary (angle 120 deg side) to keep pairs small
    a1 = np.rad2deg(np.arctan2(P1[:, 1], P1[:, 0])) % 360
    a3 = np.rad2deg(np.arctan2(P3[:, 1], P3[:, 0])) % 360
    P1 = P1[a1 < 150]; P3 = P3[a3 > 90]
    A1 = np.repeat(P1, len(P3), 0); A3 = np.tile(P3, (len(P1), 1))
    b1 = np.broadcast_to(B[1], A1.shape); b3 = np.broadcast_to(B[3], A3.shape)
    arm = seg_dist(b1, A1, b3, A3).min()
    tcp = np.linalg.norm(A1 - A3, axis=1).min()
    return arm, tcp

print("r0(mm) gap(deg) -> min_arm(mm) min_tcp(mm)")
for r0 in [100, 150, 200, 250]:
    for gap in [0, 10, 20, 30]:
        arm, tcp = worst(r0, gap)
        flag = "OK" if arm > 250 and tcp > 200 else "--"
        print(f"{r0:5d} {gap:5d} -> {arm:7.1f} {tcp:7.1f} {flag}")

def zone_pts_strip(rid, h, r0=0.0, dr=5, da=0.25):
    rs = np.arange(max(r0, 1.0), WS + 1e-9, dr)
    angs = np.deg2rad(np.arange(DIR[rid] - 60, DIR[rid] + 60 + 1e-9, da))
    R, A = np.meshgrid(rs, angs)
    P = np.c_[(R * np.cos(A)).ravel(), (R * np.sin(A)).ravel()]
    keep = np.linalg.norm(P - B[rid], axis=1) <= REACH
    for ang in (0.0, 120.0, 240.0):  # boundary rays from origin
        u = np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))])
        t = np.clip(P @ u, 0, None)
        keep &= np.linalg.norm(P - np.outer(t, u), axis=1) >= h
    return P[keep]

def worst_strip(h, r0=0.0):
    P1, P3 = zone_pts_strip(1, h, r0), zone_pts_strip(3, h, r0)
    a1 = np.rad2deg(np.arctan2(P1[:, 1], P1[:, 0])) % 360
    a3 = np.rad2deg(np.arctan2(P3[:, 1], P3[:, 0])) % 360
    P1 = P1[a1 < 160]; P3 = P3[a3 > 80]
    # thin to keep pair count manageable
    P1 = P1[::3]; P3 = P3[::3]
    A1 = np.repeat(P1, len(P3), 0); A3 = np.tile(P3, (len(P1), 1))
    b1 = np.broadcast_to(B[1], A1.shape); b3 = np.broadcast_to(B[3], A3.shape)
    return seg_dist(b1, A1, b3, A3).min(), np.linalg.norm(A1 - A3, axis=1).min(), len(P1)

print("\nstrip half-width h(mm) -> min_arm min_tcp  (zone share of workspace)")
full = np.pi * WS**2 / 3
for h in [125, 135, 150, 175]:
    arm, tcp, n = worst_strip(h)
    Z = zone_pts_strip(1, h, dr=5, da=0.25)
    # area estimate via polar cell areas
    r = np.linalg.norm(Z, axis=1); area = (r * 5 * np.deg2rad(0.25)).sum()
    print(f"h={h:4d} -> arm {arm:6.1f}  tcp {tcp:6.1f}  {'OK' if arm>250 and tcp>200 else '--'}  zone={area/full*100:4.0f}% of its sector")
