use fields::{const_fq, FieldElement, Fq, Fq12, Fq2, Fr, fq2_nonresidue};
use arith::U256;
use core::{fmt, ops::{Add, Mul, Neg, Sub}};
use rand::Rng;
use alloc::vec::Vec;

#[cfg(feature = "rustc-serialize")]
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

// This is the NAF version of ate_loop_count. Entries are all mod 4, so 3 = -1
// n.b. ate_loop_count = 0x19d797039be763ba8
//                     = 11001110101111001011100000011100110111110011101100011101110101000
//       (naf version) = 11010003030003010300300000100301003000030100030300100030030101000
// We skip the first element (1) as we would need to skip over it in the main loop
const ATE_LOOP_COUNT_NAF : [u8; 64] = [1,0,1,0,0,0,3,0,3,0,0,0,3,0,1,0,3,0,0,3,0,0,0,0,0,1,0,0,3,0,1,0,0,3,0,0,0,0,3,0,1,0,0,0,3,0,3,0,0,1,0,0,0,3,0,0,3,0,1,0,1,0,0,0];

pub trait GroupElement
    : Sized
    + Copy
    + Clone
    + PartialEq
    + Eq
    + fmt::Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + Mul<Fr, Output = Self> {
    fn zero() -> Self;
    fn one() -> Self;
    fn random<R: Rng>(rng: &mut R) -> Self;
    fn is_zero(&self) -> bool;
    fn double(&self) -> Self;
}

pub trait GroupParams: Sized + fmt::Debug {
    #[cfg(feature = "rustc-serialize")]
    type Base: FieldElement + Decodable + Encodable;
    #[cfg(not(feature = "rustc-serialize"))]
    type Base: FieldElement;

    fn name() -> &'static str;
    fn one() -> G<Self>;
    fn coeff_b() -> Self::Base;
    fn check_order() -> bool {
        false
    }
}

#[repr(C)]
pub struct G<P: GroupParams> {
    x: P::Base,
    y: P::Base,
    z: P::Base,
}

impl<P: GroupParams> G<P> {
    pub fn new(x: P::Base, y: P::Base, z: P::Base) -> Self {
        G { x: x, y: y, z: z }
    }

    pub fn x(&self) -> &P::Base {
        &self.x
    }

    pub fn x_mut(&mut self) -> &mut P::Base {
        &mut self.x
    }

    pub fn y(&self) -> &P::Base {
        &self.y
    }

    pub fn y_mut(&mut self) -> &mut P::Base {
        &mut self.y
    }

    pub fn z(&self) -> &P::Base {
        &self.z
    }

    pub fn z_mut(&mut self) -> &mut P::Base {
        &mut self.z
    }
}

#[derive(Debug)]
pub struct AffineG<P: GroupParams> {
    x: P::Base,
    y: P::Base,
}

#[derive(Debug)]
pub enum Error {
    NotOnCurve,
    NotInSubgroup,
}

impl<P: GroupParams> AffineG<P> {
    pub fn new(x: P::Base, y: P::Base) -> Result<Self, Error> {
        if y.squared() == (x.squared() * x) + P::coeff_b() {
            if P::check_order() {
                let p: G<P> = G {
                    x: x,
                    y: y,
                    z: P::Base::one(),
                };

                if (p * (-Fr::one())) + p != G::zero() {
                    return Err(Error::NotInSubgroup);
                }
            }

            Ok(AffineG { x: x, y: y })
        } else {
            Err(Error::NotOnCurve)
        }
    }

    pub fn x(&self) -> &P::Base {
        &self.x
    }

    pub fn x_mut(&mut self) -> &mut P::Base {
        &mut self.x
    }

    pub fn y(&self) -> &P::Base {
        &self.y
    }

    pub fn y_mut(&mut self) -> &mut P::Base {
        &mut self.y
    }
}

impl<P: GroupParams> PartialEq for AffineG<P> {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl<P: GroupParams> Eq for AffineG<P> {}

impl<P: GroupParams> fmt::Debug for G<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}({:?}, {:?}, {:?})", P::name(), self.x, self.y, self.z)
    }
}

impl<P: GroupParams> Clone for G<P> {
    fn clone(&self) -> Self {
        G {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl<P: GroupParams> Copy for G<P> {}

impl<P: GroupParams> Clone for AffineG<P> {
    fn clone(&self) -> Self {
        AffineG {
            x: self.x,
            y: self.y,
        }
    }
}

impl<P: GroupParams> Copy for AffineG<P> {}

impl<P: GroupParams> PartialEq for G<P> {
    fn eq(&self, other: &Self) -> bool {
        if self.is_zero() {
            return other.is_zero();
        }

        if other.is_zero() {
            return false;
        }

        let z1_squared = self.z.squared();
        let z2_squared = other.z.squared();

        if self.x * z2_squared != other.x * z1_squared {
            return false;
        }

        let z1_cubed = self.z * z1_squared;
        let z2_cubed = other.z * z2_squared;

        if self.y * z2_cubed != other.y * z1_cubed {
            return false;
        }

        return true;
    }
}
impl<P: GroupParams> Eq for G<P> {}

impl<P: GroupParams> G<P> {
    pub fn to_affine(&self) -> Option<AffineG<P>> {
        if self.z.is_zero() {
            None
        } else if self.z == P::Base::one() {
            Some(AffineG {
                x: self.x,
                y: self.y,
            })
        } else {
            let zinv = self.z.inverse().unwrap();
            let zinv_squared = zinv.squared();

            Some(AffineG {
                x: self.x * zinv_squared,
                y: self.y * (zinv_squared * zinv),
            })
        }
    }
}

impl<P: GroupParams> AffineG<P> {
    pub fn to_jacobian(&self) -> G<P> {
        G {
            x: self.x,
            y: self.y,
            z: P::Base::one(),
        }
    }
}

#[cfg(feature = "rustc-serialize")]
impl<P: GroupParams> Encodable for G<P> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        if self.is_zero() {
            let l: u8 = 0;
            l.encode(s)
        } else {
            let l: u8 = 4;
            try!(l.encode(s));
            self.to_affine().unwrap().encode(s)
        }
    }
}

#[cfg(feature = "rustc-serialize")]
impl<P: GroupParams> Encodable for AffineG<P> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        try!(self.x.encode(s));
        try!(self.y.encode(s));

        Ok(())
    }
}

#[cfg(feature = "rustc-serialize")]
impl<P: GroupParams> Decodable for G<P> {
    fn decode<S: Decoder>(s: &mut S) -> Result<G<P>, S::Error> {
        let l = try!(u8::decode(s));
        if l == 0 {
            Ok(G::zero())
        } else if l == 4 {
            Ok(try!(AffineG::decode(s)).to_jacobian())
        } else {
            Err(s.error("invalid leading byte for uncompressed group element"))
        }
    }
}

#[cfg(feature = "rustc-serialize")]
impl<P: GroupParams> Decodable for AffineG<P> {
    fn decode<S: Decoder>(s: &mut S) -> Result<AffineG<P>, S::Error> {
        let x = try!(P::Base::decode(s));
        let y = try!(P::Base::decode(s));

        Self::new(x, y).map_err(|e| match e {
            Error::NotOnCurve => s.error("point is not on the curve"),
            Error::NotInSubgroup => s.error("point is not in the subgroup"),
        })
    }
}

impl<P: GroupParams> GroupElement for G<P> {
    fn zero() -> Self {
        G {
            x: P::Base::zero(),
            y: P::Base::one(),
            z: P::Base::zero(),
        }
    }

    fn one() -> Self {
        P::one()
    }

    fn random<R: Rng>(rng: &mut R) -> Self {
        P::one() * Fr::random(rng)
    }

    fn is_zero(&self) -> bool {
        self.z.is_zero()
    }

    fn double(&self) -> Self {
        let a = self.x.squared();
        let b = self.y.squared();
        let c = b.squared();
        let mut d = (self.x + b).squared() - a - c;
        d = d + d;
        let e = a + a + a;
        let f = e.squared();
        let x3 = f - (d + d);
        let mut eight_c = c + c;
        eight_c = eight_c + eight_c;
        eight_c = eight_c + eight_c;
        let y1z1 = self.y * self.z;

        G {
            x: x3,
            y: e * (d - x3) - eight_c,
            z: y1z1 + y1z1,
        }
    }
}

impl<P: GroupParams> Mul<Fr> for G<P> {
    type Output = G<P>;

    fn mul(self, other: Fr) -> G<P> {
        let mut res = G::zero();
        let mut found_one = false;

        for i in U256::from(other).bits() {
            if found_one {
                res = res.double();
            }

            if i {
                found_one = true;
                res = res + self;
            }
        }

        res
    }
}

impl<P: GroupParams> Add<G<P>> for G<P> {
    type Output = G<P>;

    fn add(self, other: G<P>) -> G<P> {
        if self.is_zero() {
            return other;
        }

        if other.is_zero() {
            return self;
        }

        let z1_squared = self.z.squared();
        let z2_squared = other.z.squared();
        let u1 = self.x * z2_squared;
        let u2 = other.x * z1_squared;
        let z1_cubed = self.z * z1_squared;
        let z2_cubed = other.z * z2_squared;
        let s1 = self.y * z2_cubed;
        let s2 = other.y * z1_cubed;

        if u1 == u2 && s1 == s2 {
            self.double()
        } else {
            let h = u2 - u1;
            let s2_minus_s1 = s2 - s1;
            let i = (h + h).squared();
            let j = h * i;
            let r = s2_minus_s1 + s2_minus_s1;
            let v = u1 * i;
            let s1_j = s1 * j;
            let x3 = r.squared() - j - (v + v);

            G {
                x: x3,
                y: r * (v - x3) - (s1_j + s1_j),
                z: ((self.z + other.z).squared() - z1_squared - z2_squared) * h,
            }
        }
    }
}

impl<P: GroupParams> Neg for G<P> {
    type Output = G<P>;

    fn neg(self) -> G<P> {
        if self.is_zero() {
            self
        } else {
            G {
                x: self.x,
                y: -self.y,
                z: self.z,
            }
        }
    }
}

impl<P: GroupParams> Neg for AffineG<P> {
    type Output = AffineG<P>;

    fn neg(self) -> AffineG<P> {
        AffineG {
            x: self.x,
            y: -self.y,
        }
    }
}

impl<P: GroupParams> Sub<G<P>> for G<P> {
    type Output = G<P>;

    fn sub(self, other: G<P>) -> G<P> {
        self + (-other)
    }
}

#[derive(Debug)]
pub struct G1Params;

impl GroupParams for G1Params {
    type Base = Fq;

    fn name() -> &'static str {
        "G1"
    }

    fn one() -> G<Self> {
        G {
            x: Fq::one(),
            y: const_fq([
                0xa6ba871b8b1e1b3a,
                0x14f1d651eb8e167b,
                0xccdd46def0f28c58,
                0x1c14ef83340fbe5e,
            ]),
            z: Fq::one(),
        }
    }

    fn coeff_b() -> Fq {
        const_fq([
            0x7a17caa950ad28d7,
            0x1f6ac17ae15521b9,
            0x334bea4e696bd284,
            0x2a1f6744ce179d8e,
        ])
    }
}

pub type G1 = G<G1Params>;

pub type AffineG1 = AffineG<G1Params>;

#[derive(Debug)]
pub struct G2Params;

impl GroupParams for G2Params {
    type Base = Fq2;

    fn name() -> &'static str {
        "G2"
    }

    fn one() -> G<Self> {
        G {
            x: Fq2::new(
                const_fq([
                    0x8e83b5d102bc2026,
                    0xdceb1935497b0172,
                    0xfbb8264797811adf,
                    0x19573841af96503b,
                ]),
                const_fq([
                    0xafb4737da84c6140,
                    0x6043dd5a5802d8c4,
                    0x09e950fc52a02f86,
                    0x14fef0833aea7b6b,
                ]),
            ),
            y: Fq2::new(
                const_fq([
                    0x619dfa9d886be9f6,
                    0xfe7fd297f59e9b78,
                    0xff9e1a62231b7dfe,
                    0x28fd7eebae9e4206,
                ]),
                const_fq([
                    0x64095b56c71856ee,
                    0xdc57f922327d3cbb,
                    0x55f935be33351076,
                    0x0da4a0e693fd6482,
                ]),
            ),
            z: Fq2::one(),
        }
    }

    fn coeff_b() -> Fq2 {
        Fq2::new(
            const_fq([
                0x3bf938e377b802a8,
                0x020b1b273633535d,
                0x26b7edf049755260,
                0x2514c6324384a86d,
            ]),
            const_fq([
                0x38e7ecccd1dcff67,
                0x65f0b37d93ce0d3e,
                0xd749d0dd22ac00aa,
                0x0141b9ce4a688d4d,
            ]),
        )
    }

    fn check_order() -> bool {
        true
    }
}

pub type G2 = G<G2Params>;

pub type AffineG2 = AffineG<G2Params>;

#[cfg(test)]
mod tests;

#[test]
fn test_g1() {
    tests::group_trials::<G1>();
}

#[test]
fn test_g2() {
    tests::group_trials::<G2>();
}

#[test]
fn test_affine_jacobian_conversion() {
    let rng = &mut ::rand::thread_rng();

    assert!(G1::zero().to_affine().is_none());
    assert!(G2::zero().to_affine().is_none());

    for _ in 0..1000 {
        let a = G1::one() * Fr::random(rng);
        let b = a.to_affine().unwrap();
        let c = b.to_jacobian();

        assert_eq!(a, c);
    }

    for _ in 0..1000 {
        let a = G2::one() * Fr::random(rng);
        let b = a.to_affine().unwrap();
        let c = b.to_jacobian();

        assert_eq!(a, c);
    }
}

#[inline]
fn twist() -> Fq2 {
    fq2_nonresidue()
}

#[inline]
fn two_inv() -> Fq {
    const_fq([
        9781510331150239090,
        15059239858463337189,
        10331104244869713732,
        2249375503248834476,
    ])
}

#[inline]
fn twist_mul_by_q_x() -> Fq2 {
    Fq2::new(
        const_fq([
            13075984984163199792,
            3782902503040509012,
            8791150885551868305,
            1825854335138010348,
        ]),
        const_fq([
            7963664994991228759,
            12257807996192067905,
            13179524609921305146,
            2767831111890561987,
        ]),
    )
}

#[inline]
fn twist_mul_by_q_y() -> Fq2 {
    Fq2::new(
        const_fq([
            16482010305593259561,
            13488546290961988299,
            3578621962720924518,
            2681173117283399901,
        ]),
        const_fq([
            11661927080404088775,
            553939530661941723,
            7860678177968807019,
            3208568454732775116,
        ]),
    )
}

#[derive(PartialEq, Eq)]
pub struct EllCoeffs {
    pub ell_0: Fq2,
    pub ell_vw: Fq2,
    pub ell_vv: Fq2,
}

#[derive(PartialEq, Eq)]
pub struct G2Precomp {
    pub q: AffineG<G2Params>,
    pub coeffs: Vec<EllCoeffs>,
}

impl G2Precomp {
    pub fn miller_loop(&self, g1: &AffineG<G1Params>) -> Fq12 {
        let mut f = Fq12::one();

        let mut idx = 0;

        for i in ATE_LOOP_COUNT_NAF.iter() {
            let c = &self.coeffs[idx];
            idx += 1;
            f = f.squared()
                .mul_by_024(c.ell_0, c.ell_vw.scale(g1.y), c.ell_vv.scale(g1.x));

            if *i != 0 {
                let c = &self.coeffs[idx];
                idx += 1;
                f = f.mul_by_024(c.ell_0, c.ell_vw.scale(g1.y), c.ell_vv.scale(g1.x));
            }
        }

        let c = &self.coeffs[idx];
        idx += 1;
        f = f.mul_by_024(c.ell_0, c.ell_vw.scale(g1.y), c.ell_vv.scale(g1.x));

        let c = &self.coeffs[idx];
        f = f.mul_by_024(c.ell_0, c.ell_vw.scale(g1.y), c.ell_vv.scale(g1.x));

        f
    }
}

pub fn miller_loop_batch(g2_precomputes: &Vec<G2Precomp>, g1_vec: &Vec<AffineG<G1Params>>) -> Fq12 {
    let mut f = Fq12::one();

    let mut idx = 0;

    for i in ATE_LOOP_COUNT_NAF.iter() {
        f = f.squared();
        for (g2_precompute, g1) in g2_precomputes.iter().zip(g1_vec.iter()) {
            let c = &g2_precompute.coeffs[idx];
            f = f.mul_by_024(c.ell_0, c.ell_vw.scale(g1.y), c.ell_vv.scale(g1.x));
        }
        idx += 1;
        if *i != 0 {
            for (g2_precompute, g1) in g2_precomputes.iter().zip(g1_vec.iter()) {
                let c = &g2_precompute.coeffs[idx];
                f = f.mul_by_024(c.ell_0, c.ell_vw.scale(g1.y), c.ell_vv.scale(g1.x));
            }
            idx += 1;
        }
    }

    for (g2_precompute, g1) in g2_precomputes.iter().zip(g1_vec.iter()) {
        let c = &g2_precompute.coeffs[idx];
        f = f.mul_by_024(c.ell_0, c.ell_vw.scale(g1.y), c.ell_vv.scale(g1.x));
    }
    idx += 1;
    for (g2_precompute, g1) in g2_precomputes.iter().zip(g1_vec.iter()) {
        let c = &g2_precompute.coeffs[idx];
        f = f.mul_by_024(c.ell_0, c.ell_vw.scale(g1.y), c.ell_vv.scale(g1.x));
    }
    f
}

#[test]
fn test_miller_loop() {
    use fields::Fq6;

    let g1 = G1::one()
        * Fr::from_str(
            "18097487326282793650237947474982649264364522469319914492172746413872781676",
        ).unwrap();
    let g2 = G2::one()
        * Fr::from_str(
            "20390255904278144451778773028944684152769293537511418234311120800877067946",
        ).unwrap();

    let g1_pre = g1.to_affine().unwrap();
    let g2_pre = g2.to_affine().unwrap().precompute();

    let gt = g2_pre.miller_loop(&g1_pre);

    let expected: Fq12 = Fq12::new(
            Fq6::new(
                Fq2::new(
                    Fq::new(U256([51910954035973319022896381997847359481, 49070349125448662928383548013678560320])).unwrap(),
                    Fq::new(U256([150594250655925940766158230906714822921, 45067780486977162411874315270532662559])).unwrap(),
                ),
                Fq2::new(
                    Fq::new(U256([293313211826787380313097274184299135668, 28033688961864567415258173424862015279])).unwrap(),
                    Fq::new(U256([167463228417728651969785007140185669229, 7077084888790581611350259269763958251])).unwrap(),
                ),
                Fq2::new(
                    Fq::new(U256([166574695108782631900870170909221310910, 36301755601680728879208628452507017454])).unwrap(),
                    Fq::new(U256([61790765844042689836493058059938629070, 8459680572251855304146082351314233167])).unwrap(),
                )
            ),
            Fq6::new(
                Fq2::new(
                    Fq::new(U256([274725556782132290265566702453516000786, 47645385003117491559484060631887523335])).unwrap(),
                    Fq::new(U256([218741759704184655717004970623859820160, 5768209145436844234600983552836237590])).unwrap(),
                ),
                Fq2::new(
                    Fq::new(U256([166365676746880051357185694330614395245, 44422629177536239628987108174157680084])).unwrap(),
                    Fq::new(U256([188797990739833756731082975171894736944, 643465180603364587407484249282263717])).unwrap(),
                ),
                Fq2::new(
                    Fq::new(U256([271144479861903489720584548513988144824, 10463758518630442972881156820224659715])).unwrap(),
                    Fq::new(U256([214759070354702766397810519515686065785, 63150584453541665372008601383729030318])).unwrap(),
                )
            )
        );
    assert_eq!(gt, expected);
}

impl AffineG<G2Params> {
    fn mul_by_q(&self) -> Self {
        AffineG {
            x: twist_mul_by_q_x() * self.x.frobenius_map(1),
            y: twist_mul_by_q_y() * self.y.frobenius_map(1),
        }
    }

    pub fn precompute(&self) -> G2Precomp {
        let mut r = self.to_jacobian();

        let mut coeffs = Vec::with_capacity(102);

        let q_neg = self.neg();
        for i in ATE_LOOP_COUNT_NAF.iter() {
            coeffs.push(r.doubling_step_for_flipped_miller_loop());

            if *i == 1 {
                coeffs.push(r.mixed_addition_step_for_flipped_miller_loop(self));
            }
            if *i == 3 {
                coeffs.push(r.mixed_addition_step_for_flipped_miller_loop(&q_neg));
            }
        }
        let q1 = self.mul_by_q();
        let q2 = -(q1.mul_by_q());

        coeffs.push(r.mixed_addition_step_for_flipped_miller_loop(&q1));
        coeffs.push(r.mixed_addition_step_for_flipped_miller_loop(&q2));

        G2Precomp {
            q: *self,
            coeffs: coeffs,
        }
    }
}

impl G2 {
    fn mixed_addition_step_for_flipped_miller_loop(
        &mut self,
        base: &AffineG<G2Params>,
    ) -> EllCoeffs {
        let d = self.x - self.z * base.x;
        let e = self.y - self.z * base.y;
        let f = d.squared();
        let g = e.squared();
        let h = d * f;
        let i = self.x * f;
        let j = self.z * g + h - (i + i);

        self.x = d * j;
        self.y = e * (i - j) - h * self.y;
        self.z = self.z * h;

        EllCoeffs {
            ell_0: twist() * (e * base.x - d * base.y),
            ell_vv: e.neg(),
            ell_vw: d,
        }
    }

    fn doubling_step_for_flipped_miller_loop(&mut self) -> EllCoeffs {
        let a = (self.x * self.y).scale(two_inv());
        let b = self.y.squared();
        let c = self.z.squared();
        let d = c + c + c;
        let e = G2Params::coeff_b() * d;
        let f = e + e + e;
        let g = (b + f).scale(two_inv());
        let h = (self.y + self.z).squared() - (b + c);
        let i = e - b;
        let j = self.x.squared();
        let e_sq = e.squared();

        self.x = a * (b - f);
        self.y = g.squared() - (e_sq + e_sq + e_sq);
        self.z = b * h;

        EllCoeffs {
            ell_0: twist() * i,
            ell_vw: h.neg(),
            ell_vv: j + j + j,
        }
    }
}

#[test]
fn test_prepared_g2() {
    let g2 = G2::one()
        * Fr::from_str(
            "20390255904278144451778773028944684152769293537511418234311120800877067946",
        ).unwrap();

    let g2_p = g2.to_affine().unwrap().precompute();

    let expected_g2_p = G2Precomp {
        q: AffineG {
            x: Fq2::new(
                Fq::new(U256([286108132425575157823044300810193365120, 40955922372263072279965766273066553545])).unwrap(),
                Fq::new(U256([51787456028377068413742525949644831103, 18727496177066613612143648473641354138])).unwrap()
            ),
            y: Fq2::new(
                Fq::new(U256([105235307093009526756443741536710213857, 56697136982397507595538316605420403515])).unwrap(),
                Fq::new(U256([329285813328264858787093963282305459902, 40620233227497131789363095249389429612])).unwrap()
            )
        },
        coeffs: vec![
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([145915152615018094207274265949237364577, 7720188347992263845704223037750674843])).unwrap(), Fq::new(U256([1642602221754736777334297091439332137, 43455737427254701713230610624368790806])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([192300176042178641247789718482757908684, 15253255261571338892647481759611271748])).unwrap(), Fq::new(U256([84481530492606440649863882423335628050, 47407062771372090504997924471673219553])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([232941999867492381013751617901190279960, 33161335727531633874118709394498248694])).unwrap(), Fq::new(U256([205159091107726234051689046255658875895, 18784742195738106358087607099254122817])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([44903717410722925336080339155272113598, 2432148164440313442445265360131108750])).unwrap(), Fq::new(U256([115058944839679151514931430187558137193, 19913547532675326995005163334665928687])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([162225342243516818284353803749727397649, 2902812998987289816105769597190405892])).unwrap(), Fq::new(U256([280882305470180330735174149091034017688, 45000755139204939118391532933575113126])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([81439170073299243743458979294198176188, 8269567577349210062003854855333109246])).unwrap(), Fq::new(U256([229588642195598894684486216729209468602, 50120631775179825949138513731773141671])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([62571086408084349816772432183764211635, 50233214695378430835319156584725532968])).unwrap(), Fq::new(U256([187531012616569709709565459550531626708, 25399966597901061874032352675540749982])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([125202410210363842487043120679844927964, 32132029049461004568346717732357115368])).unwrap(), Fq::new(U256([40047990517661970904320150752358301419, 29547330482153702904408059941549019061])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([72725441636162608820485881978479394755, 4335829118654814821890758647864621059])).unwrap(), Fq::new(U256([3808530996119731737716737121624391430, 14840119309897919310042357130477699760])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([230923838046707456761202857863657685779, 30663581424308030115949734185266015957])).unwrap(), Fq::new(U256([75804784823545649980980649059385057066, 40783805432413650791639865054248573254])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([220403174695454806701510356109644664970, 61694366506643321007589345694791625235])).unwrap(), Fq::new(U256([111050616308506297168851165107648527276, 29953858315513175963460036136567176100])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([339360714692284035291265180647668192278, 51066683198194726198561068977822786062])).unwrap(), Fq::new(U256([20532264324124291215869821260466642056, 53772344434441015474139364759811842825])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([53390632401876602131528763251836289052, 7039696239984002955384050250007606842])).unwrap(), Fq::new(U256([120756148133186395786471763371515340851, 62560956052144901655506280432592733452])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([308576049186755490218939550330371718607, 60158766280438165965298754059752443327])).unwrap(), Fq::new(U256([36519660971413271753359371158311247925, 33166369001790145692925310192829466902])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([178902296336418821251231963767960641976, 54039139529656231144797458475123200306])).unwrap(), Fq::new(U256([327709063289359470903820254941214044104, 29921523827667065809753199129798632038])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([293947345738284310028696202770439717312, 23629199470233866978708839571836323911])).unwrap(), Fq::new(U256([270255405451020462647135094276398320650, 10214498461032217021941670870340839535])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([159737597162744224577379609113239040296, 42414292048147664393527940182644080820])).unwrap(), Fq::new(U256([319166235401593060755720596880844332847, 23942967675619615172318074973192451617])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([322582854267025287594765649802803266076, 43815838906866149475744535388480851056])).unwrap(), Fq::new(U256([194731262617323725381582694441938082298, 6318043913263416116808511289860376787])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([340258202167813930211448229264749672038, 49309561646299510355653200516855272117])).unwrap(), Fq::new(U256([314009511129121341897071879080163927420, 46733418159091100849346822998375177924])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([296127326349185873181435771941022502705, 27092339993712929447599322475752229815])).unwrap(), Fq::new(U256([14090774041595571810302865485069892149, 9416959189726308679959984066033622764])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([331192186309261416893996962356905314610, 6133490180262551399580874298514378708])).unwrap(), Fq::new(U256([52146008553438513232574426823996261862, 51692389190081578424424830745703412620])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([304886724958645105936240400377782311329, 24452949520921718326419149662293231897])).unwrap(), Fq::new(U256([191644523888771552477578834463319391872, 21762407072532056308389337796258587948])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([131875657381595539531276734057336373872, 46145756403556903236477737806904884925])).unwrap(), Fq::new(U256([15513558811446045405402043661936649706, 6164660913794769127674854991137805768])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([37628908878688989915703006228108158588, 32333568458099455743134394992114761192])).unwrap(), Fq::new(U256([310028653933656868572053471966984499925, 50927285178735796712220443510555951806])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([323579290279633699483736616036574636320, 26462458255852549637810201830373030787])).unwrap(), Fq::new(U256([212356178148004645362110693008071812466, 7147771462841335914859391157142419270])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([33793486114682505251748396905247663179, 12971208030293312519168903516345112146])).unwrap(), Fq::new(U256([105470215179403498779276452125929471612, 13243999928262666388833873861601164323])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([336774965397927402479055329499391144750, 14092422716202439890370586466555281245])).unwrap(), Fq::new(U256([128524990965480598411024491536330888971, 16339349487558512101009117692791080358])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([308807709363499650659552856561459119304, 20549512613498060437905021698882254003])).unwrap(), Fq::new(U256([296831825878801592442122280385770051837, 12999356604250035352468537408632919011])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([6416952549311962005690662486801715420, 6562024922514861810686738146979657369])).unwrap(), Fq::new(U256([208034701293282569885312328681897330015, 32565296199581814532888690261728578093])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([158821090683358871468006667057662171383, 43639936137708212270934870661514052516])).unwrap(), Fq::new(U256([273462267971100585683746592136866263017, 33686004986452254374790215788772597950])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([239050458738191424179259026508283200850, 59033062634877072212375745660456082761])).unwrap(), Fq::new(U256([42735232027681689333255156554135804585, 3645324355274008091830665999900460711])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([108736755942765297679539162065532976047, 53005795603772553534072950762621113576])).unwrap(), Fq::new(U256([161143215360647768117453097687908448132, 47995551574176504820342605188594172079])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([6657347986892301669776633258074701543, 11902481233202623619409687758723915636])).unwrap(), Fq::new(U256([222956595276083444090603007056080975801, 36382879863546362219365726153085767156])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([313682093040115002688525404999246872446, 42956501626298358982264420981703353471])).unwrap(), Fq::new(U256([16910327688907897736932577003378075753, 1359671211117678708781834746493106650])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([6643221945930263241861939865579251413, 53936065521860576156419486432192485437])).unwrap(), Fq::new(U256([309056291241266457640134736864478816427, 37247957207987268075661977484645707642])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([222157301260856896664619946160442713805, 49516872999212369072183141842522894943])).unwrap(), Fq::new(U256([205513791444704603844975928062124661076, 57468270224470125473802906477068197830])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([8285963000933560101573593109316673578, 14786122523410515396563958444109027710])).unwrap(), Fq::new(U256([122158657553570962061566147876423950513, 52139940048286310424786112221915773798])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([334881138980927901869290199603379387015, 25426740963889929124688873103192284934])).unwrap(), Fq::new(U256([15041987314727689102877759056668987416, 53279076961030137249765180937167164331])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([219851721339103004718273461131600868626, 36871882814964809764289908414024625695])).unwrap(), Fq::new(U256([131269942067631553063094618567097759738, 53556533347945713183394135477679452243])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([303322002694907044040904386256578801369, 47818402373691165133570276489205539247])).unwrap(), Fq::new(U256([23840328058016663665484464556433078317, 57137757685631244994429644684135564308])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([313347268925459844355339673983448992252, 63059125438530014597733264556246943889])).unwrap(), Fq::new(U256([332951624867081114956044501931839262097, 21446965953220512177611603903918604526])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([43308962429154740283739513584277671586, 36726788540078024416606994278261486896])).unwrap(), Fq::new(U256([201462004284693135670392448782496773481, 37361169637624833127302583281977615660])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([23765318454509223469921591210222962901, 44990033695398054388153021046567582164])).unwrap(), Fq::new(U256([31305053791054889490552967007711680166, 27103171064041391806294729422128985527])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([276126480562835962332188871122724756846, 16736136826306680287985573076016202896])).unwrap(), Fq::new(U256([4414319697794720805306194236550991584, 62019193108561650400059767099940339027])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([264396677944255651896935058261233808058, 56759185289081036894005736531633938393])).unwrap(), Fq::new(U256([203628704629834831927134686796555271069, 48471469187143841926703236705753907861])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([239115594351165984647685996782641428320, 44545149465284571473177119694335581490])).unwrap(), Fq::new(U256([213078788930221853484251159402089402698, 16763897914971151155224113448725254022])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([114008097138551618719252989046711208100, 59881147407771812558336453348641148192])).unwrap(), Fq::new(U256([73126778233281675579715222201222232223, 38826300453213723195010574430400377747])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([241042448915505783580706972981273226980, 56417003146210087305012985177072964806])).unwrap(), Fq::new(U256([169177487653248888684114813249577243086, 32380354283087538368804702239086564787])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([158019800920508883600584012748066500495, 26898142140797843980908437809386681141])).unwrap(), Fq::new(U256([306502797874464726523541130582372927512, 24906313146242057911294566839234262300])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([285760763932996537001129740504018931266, 61805280449958461312688755177468072593])).unwrap(), Fq::new(U256([135545131264687967800977791145788692523, 58139772512775079850670752006123037712])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([70329035009259499088740735594160571447, 39714506367478198523957872987679138518])).unwrap(), Fq::new(U256([140879819390078281356249388675648473246, 43854973736398482797037510623636581696])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([38618198701391964654224360704900902096, 38909748519501972274789640400408937483])).unwrap(), Fq::new(U256([180923031305898977193921630254659948144, 27460338595509200370239569387614912648])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([70057505024221161194312385639309591716, 18962666975131076741596858262806841593])).unwrap(), Fq::new(U256([214987899942827030987983709303642547609, 7354730872853772995022889424236171557])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([915509610432328048899540473086774484, 34629556430715177067961193484853677166])).unwrap(), Fq::new(U256([210548802885276746342162088554780886285, 59946451357630926783981469963615555141])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([192587880731165909190755427373555634019, 10976459312692912490911357520717353871])).unwrap(), Fq::new(U256([167196051583193420579377374296283755878, 49401460467894024879692901111127305932])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([280565789968090605174946518022513606221, 37294483328987043033011583201807512274])).unwrap(), Fq::new(U256([43162969061744994465193287354736460751, 22121767444104992380234797550707180209])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([253802905125777094707425424576407998829, 61686094369188045360896737944218403088])).unwrap(), Fq::new(U256([182974256593102716284343824242596548794, 7848028085342627691056691434137748535])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([84344031984214336466355469526466525167, 27093607600401090432285884248864703045])).unwrap(), Fq::new(U256([149328956099164822232925589118520984554, 2238105695079930209978670615986667957])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([164524345257317963111443719298495629688, 60976929030698527695079882388987310770])).unwrap(), Fq::new(U256([306918226890341242065585311833805224781, 55722896155980110701397483880645840750])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([93128913154066328529254602049518310767, 33148637643432199376041913783211199036])).unwrap(), Fq::new(U256([232426184416524951250597496934702636450, 43720696322889678655980069182616949616])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([102654491688396809766739755399745558766, 38605113922396003219627575078289843591])).unwrap(), Fq::new(U256([19593962899181063954599229676070513364, 60350965404501231355169027639339453296])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([159636507003615123705820979675398042179, 54313579941440479276773225578829814974])).unwrap(), Fq::new(U256([119356164563490256235392395335237959869, 39953340372362730589797197026608570557])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([63485981942258213001157530320762394972, 31200716913018312733656963741875248687])).unwrap(), Fq::new(U256([12902614517986449375080893856817237357, 18985985729109591831690040180758616083])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([42308655177239821867892255001741444297, 48752963962065714616191625548108808])).unwrap(), Fq::new(U256([128521931437576558829282023068349026555, 41085812991756679343188097975462933419])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([10035402786640614748259945360125292132, 38557266307605033249489727552659930268])).unwrap(), Fq::new(U256([261287166540169381393014929830322479654, 42514049292646924562828996896910004962])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([56319885731071444245573249522306017699, 53577564523791470572591472840806985932])).unwrap(), Fq::new(U256([124520094647080351085028512262335797477, 34908241061842840630524194497782012074])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([126634618253919797596379978688751079772, 52145187142657016821034369446677208400])).unwrap(), Fq::new(U256([331622287147942589908508350657315302543, 33514242158733995981408830992771427380])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([120940515437555419164907643421042686361, 17509271053436215565086784812667241751])).unwrap(), Fq::new(U256([115780741523467210253488502554959480585, 32352495928341192491861819481034439523])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([3619315753849520875056567968398971518, 24276420029996570458336249792816120452])).unwrap(), Fq::new(U256([211494135803191326215038551543926917828, 43184746342435212339590913052197238451])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([299074873552726234274845052736068685064, 17100980781941753342716208137755315141])).unwrap(), Fq::new(U256([132717845842422445158092928854959930670, 44970515850781790514922204901459656647])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([216867730679061124743178802618784487478, 57695717306148677588162001948381364803])).unwrap(), Fq::new(U256([270218131380365097755408360718075538527, 57477676726794026534017337223493668494])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([165279544895097377741304748618110436848, 20425223618131822879513743747480509436])).unwrap(), Fq::new(U256([74939006713979852471760437945055491248, 13009460391442509409368588074771283312])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([66759827770349370888034213457290956508, 9005955800938032302448930728853089346])).unwrap(), Fq::new(U256([132741330683696165661450722498530220360, 7366906791443646993778243377846220425])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([253540438895289112523278690834655986794, 6116090159416232554407405788526490758])).unwrap(), Fq::new(U256([30829083910163512740218467627901622426, 55620601051021580444864016142195538972])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([250454368142865329167214266233567636554, 62357304312342518984982585510065736692])).unwrap(), Fq::new(U256([188759902529624961292182168219378957599, 46090313440960353890072197786322881372])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([273524456438634570870477083331972978477, 9134704483036112456254510849911524401])).unwrap(), Fq::new(U256([67478468272116511526567438494812739067, 25741563822514337717816912706935692500])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([116593352960394465976476340555861894686, 41112534697095493718894382902662572752])).unwrap(), Fq::new(U256([120743240887113383410964567068981375379, 38598788085595749947162720656886407615])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([312187803926537773957821532171857044863, 19370576499152094014450666672812249236])).unwrap(), Fq::new(U256([240827398452930199439299722869808431504, 49509218268486878713361004562421178491])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([290888165778751654611450374067314903262, 37392757737237359968235356696866333927])).unwrap(), Fq::new(U256([252204440608865557722718990121112495320, 55975286037277175922442767802285953079])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([270039432299546232993592658461203399951, 60656209400809596126720306545905143183])).unwrap(), Fq::new(U256([285724268091089729569589544862374658568, 31963711474983717334851248248346512047])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([3884505529011790540935411500477268151, 13223063167340752683657032423528371760])).unwrap(), Fq::new(U256([159305752483642939978115090933392332739, 7151624608160356980135319367119889070])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([272676514700352082082833454466828720931, 25063412692149354657238579769317314943])).unwrap(), Fq::new(U256([293621566718932528694833049559922653569, 45585304922174719737960365050252747127])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([54835393883782456964222308618682542993, 37980634195943561261187934261099825702])).unwrap(), Fq::new(U256([59974402342646963556531661011524247325, 35696051147487701670582425632191066303])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([259456005420154607862207518414346338959, 26130920677564288823895676956971235383])).unwrap(), Fq::new(U256([116130095060524137327607346021153250193, 10807286360635090367088053443993301396])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([49392952615759746255132704409838115781, 47072336582571607445080612120532993304])).unwrap(), Fq::new(U256([176567394438796320529775330793638953568, 28322815468520297894419530294472748019])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([80922979608932498062547214350892680237, 39454405473363814575369344229244849847])).unwrap(), Fq::new(U256([91401220192762005776280059066692026476, 15657465297119889794043496656394015039])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([66039352326985904039959490473173702089, 53019684969492620390541557616482571566])).unwrap(), Fq::new(U256([241714503488139707849386627025007386925, 26075944201145781355709956722513636060])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([282595284318374809009793390206732488298, 45748314157085313197379566607025256839])).unwrap(), Fq::new(U256([270827203295353120306891953779231995484, 2061164657869992110070755346638481586])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([122772591180024083766405253289036218458, 22573428243213343555967649475581629995])).unwrap(), Fq::new(U256([335389040951271989935196787085438096490, 19801736168278388699669751501679386348])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([339013185026928567100232707644090060309, 19772463825247091662201186215246641510])).unwrap(), Fq::new(U256([4804384610649622501550615795150497154, 23912076036945346828195974346279787457])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([214353258010296277687883233632193811097, 5219474783704014986415232032947372339])).unwrap(), Fq::new(U256([79624479932762034850342328421303227111, 56787485322265889115274972394242117773])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([184200025840089548499823017798379240877, 48381517011877532244879858559402925202])).unwrap(), Fq::new(U256([107565933978602112957892502247036378430, 45988606090744912079186414974711188065])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([110593764674444376346243358175616189408, 35712801833060284055338218769830573236])).unwrap(), Fq::new(U256([151909254176585476474604484796159407774, 47158276852552318249500199498589330637])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([322742519015404082194067614300530448854, 59892171075434557254836830922854923608])).unwrap(), Fq::new(U256([32065386578298932256366582168915019527, 26752577181504287044232901657950904669])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([174803672897784440584915958306641383601, 41091446500573741637560530719439634516])).unwrap(), Fq::new(U256([209867082646016733058415245870200692364, 62329211607238784395268250532556435532])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([306258598889945544178464082933019504953, 54721033386841216374878966418312958051])).unwrap(), Fq::new(U256([60603024880089835430212078876452798438, 46149383487322374768616169869877411190])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([217026022228859342220342856144348485905, 57413523264998851221438276699831981764])).unwrap(), Fq::new(U256([124187198815039194427087238512103681551, 5443344443330946400667796667492661757])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([329826946228202959132718054863501702187, 48520872455619090037740041830322500851])).unwrap(), Fq::new(U256([84830400552123989859098307569998867522, 48734171266537512281517863759978814030])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([223121539165880683751973996773385673012, 25883364325220384256766392563193579368])).unwrap(), Fq::new(U256([299825902106679285294937343573961127742, 60789713632874624396137039249203910802])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([256516074149138488025765885988857744378, 14065986710590565469822003697004376208])).unwrap(), Fq::new(U256([234079806620020815932181140203001517322, 23367862088314049165195572485764983176])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([196437028129330799836797952742061629213, 48672430434915053819319219832418834574])).unwrap(), Fq::new(U256([325437437809364731749456582922626432887, 48166506837989600793252437510600468487])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([159308630334191903727834955555702331135, 62436879262294424125843661083177310796])).unwrap(), Fq::new(U256([36766573815532717355926225449599862176, 45422312033048180700369981630678147962])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([242582781609674647644149440448686667005, 38749506523024653240132899426202698768])).unwrap(), Fq::new(U256([175781266358339061379292613713370680613, 18670420110220807116103915283744134837])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([330308428247032904787707230285980997411, 57858504610626027326513601505745660271])).unwrap(), Fq::new(U256([137930876960001856905683673437505764610, 51655613850377410017968900339454298922])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([313120701922067853802383887994976410392, 45975781155280265622735990880466050470])).unwrap(), Fq::new(U256([335042362116388769047290965043326509112, 60559196975175152299532260702584038813])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([168129439507943874458360855948848921748, 49297302702020077848733359456178701712])).unwrap(), Fq::new(U256([269089028431861389215030548077466532939, 39860705713699087717989916919780377864])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([13752419440807207645064267622022228455, 58622973202646179187733854793315255353])).unwrap(), Fq::new(U256([214305013520593573583121266790878302778, 38633624705329558986475404541639527124])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([149534517263071862076005511513725705381, 33923419970735724678578488530397069094])).unwrap(), Fq::new(U256([52240222008682291290000393139971676649, 13553337484321113195355299279586301175])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([272900340425393991345320871434898164480, 38006376657960211098766747755417992280])).unwrap(), Fq::new(U256([261987172512287267237176115671442326513, 8128050373266945066047548013741537435])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([262663552198367871113333688806502970707, 16999213382693802141971886256404278345])).unwrap(), Fq::new(U256([82246501115823748400993076962512494264, 9513280785331020628797944368204762745])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([68225301292307439157467124956340428751, 23984526331358356113154550150873972672])).unwrap(), Fq::new(U256([322267803776402446716363035932937663522, 7643551605057856674267205902314118400])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([226851502497426762485425012577563658166, 33899046058167810318161894461593654070])).unwrap(), Fq::new(U256([106137231667287271021611800141669940221, 38201180579053106362613109983520968531])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([75276265667487657641526123222570637963, 18790008468742777802887427569490235644])).unwrap(), Fq::new(U256([236348579709234796216705203173042603149, 51488118889105930334369568118204095717])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([70847827090593073015786004614483673839, 3475230874891506896203634678143642692])).unwrap(), Fq::new(U256([8314731801106602343291488634630259831, 3521008420700564106163167660007668706])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([92354268952660546820822603858288251059, 19778715140389467112147527814081030657])).unwrap(), Fq::new(U256([329385425284714431570594794971018721246, 60587846563454266929636649850787073194])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([288633840471857542459144118014949000217, 51605528849039220650036727626483827846])).unwrap(), Fq::new(U256([251176740648089198193669671101929833609, 48200613470991338988108533743364275499])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([329193234492690486040808567065220826749, 21378867659438236011740749197423248550])).unwrap(), Fq::new(U256([285725508694128806244839238836446595834, 16111887615812423492270406970991867951])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([152431519506548878061194990588057484057, 49184327280433466363535994383523121304])).unwrap(), Fq::new(U256([176897083100713735838942116510311567986, 37010995217032206230292299420436992452])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([130722401240365261778198584122678322362, 19293082368638790806988358431355209633])).unwrap(), Fq::new(U256([336032502313799023638100083369289206306, 31063427179065887680551186789065013379])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([313119215174514058940660103265063450959, 25103967051083955792570154250792420707])).unwrap(), Fq::new(U256([31520718700780462452153016807890400365, 3961179668062303572693429066523266858])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([36667789591031625960240713279402427923, 5637784774941607531109129433890086293])).unwrap(), Fq::new(U256([286537786902526732905389646221547183400, 56733382321156059969864849624810202662])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([318803650206508773198520172514605937449, 26727846312556415159035648438885744059])).unwrap(), Fq::new(U256([296466727057750056890238712141155450998, 27938872020780344351123790536034462604])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([324800882000687209140271724886698306645, 44039571306759586933457684739133508444])).unwrap(), Fq::new(U256([47608245752536926992583482090528259005, 47831283599118732387313994410846427415])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([234409386597337888556308771844488457344, 12956567414802650084616113864217789370])).unwrap(), Fq::new(U256([54897214305565100948138138335662741789, 15577398627840234231085370089137424292])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([133034516621699526624516176710921958663, 21900941110792896857560826682314104825])).unwrap(), Fq::new(U256([91786723574963827227124861422083359642, 15631994703072556950984703411138300334])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([61994842824519647915165015802348883298, 3606506510538278544457628965895793792])).unwrap(), Fq::new(U256([176676927874382884508332121131239871282, 40115823781693004388411447312641595826])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([16486756791893393092802670694010735350, 51213785425023838862657045286249924761])).unwrap(), Fq::new(U256([127374475998848055784786954745585192546, 61927128032062116041960763307324749312])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([57390628646817442883586335851238696952, 1896752620999083872406878608493454725])).unwrap(), Fq::new(U256([221112496886805666932828220912733522283, 2796002105546343313342619098780470642])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([126113866998638086000522716026151108800, 58788515162206661637618804912085907736])).unwrap(), Fq::new(U256([19079621171649876430748555077055696995, 46826087543975014201271501761421031441])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([287527947651847089840085422221128698763, 3282429064220775329203352366444259855])).unwrap(), Fq::new(U256([307065141447870398567299540185758941637, 48193280250277647112409463651594786821])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([39874018308944266937776827490734641649, 37791912819895325896728519088674532352])).unwrap(), Fq::new(U256([11619188332176047954604673895207823813, 12983193006960451462302777892755566431])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([76045219079383487030610001932731241273, 16472513195230803643409094386807276517])).unwrap(), Fq::new(U256([244195048886093761135684126358283143738, 19479715688038787901591036330200598003])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([10563934263428977131610186732868968998, 14179527485555674710937849675581267906])).unwrap(), Fq::new(U256([257836314204976892854269567329818300595, 50763620022812667001295845381267741649])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([160125271831790681583846356333205971645, 38979573670559551140762745448476278558])).unwrap(), Fq::new(U256([258822435332825467396267733276444939733, 19361127335094998017975714923507638644])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([5086484109635987997951722709219418410, 11993539151823720275736182499510363133])).unwrap(), Fq::new(U256([171423009691627362399764794909459189507, 20404136165000279368092981943358482669])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([249500614137833023351305910264605008487, 42310425481082032237299673828774704531])).unwrap(), Fq::new(U256([253496613223900195266106558972296162241, 19046240361322320039639197826388768369])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([72263736846331351747703353269055941107, 19969702258117494907412570199791763548])).unwrap(), Fq::new(U256([126173921786290239044307538655562024455, 18006158804314393556918633600879748967])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([16201000244836956196708576962189993382, 32135871927111623648702296852432295139])).unwrap(), Fq::new(U256([186963801770492707056456887595022261993, 2392014174637078162701730926122882282])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([8922998024015454766590940715881630670, 55567280047347356367358986337109964651])).unwrap(), Fq::new(U256([313173491437683437471834863194199707492, 3438176908844635046641649505356812159])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([276513541262028309851175912267910407247, 11212571122829858982221117734299851084])).unwrap(), Fq::new(U256([259509606552160006313745402323745013955, 43114059504717558659877851383190095613])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([28098698558455079499553496395853067214, 2999821831816440733771768849050005165])).unwrap(), Fq::new(U256([334260605709723946550826958885726232933, 11843826148467149350077605819950881153])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([283455530950531157108702282345810919101, 52521643621580203161269969006017546759])).unwrap(), Fq::new(U256([118200123115287742628646618978307449884, 14795263733303838767894982171654042612])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([132641709504834974317245159550618583253, 28570895840581787308435502085893978558])).unwrap(), Fq::new(U256([229949535634549796967397304164923056337, 42352202734287711800564772982472784839])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([128199924683494278990710814519750517837, 58638050185358930355922065483565476451])).unwrap(), Fq::new(U256([4192016302378839786647752861109219112, 61114929591783829703998838288847230811])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([113754079486581376183180167710451964178, 25998717875687967363117010408717972365])).unwrap(), Fq::new(U256([133697001740616163002906224555636440440, 23546896556625148713807775573505304385])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([227150749988479615370826356819617461792, 41368715815045829029754124051658999245])).unwrap(), Fq::new(U256([222514455614853854819390285868145864687, 27579121989432504016710179821357885225])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([128433415390196263739951350076595027296, 55636438757722652255615208065796848545])).unwrap(), Fq::new(U256([94671524825895380554049635623924644741, 9791488862788864605119131922310440121])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([153881172058568095361353223261899310845, 17156484263879687903526208855029043640])).unwrap(), Fq::new(U256([192077265925835218260989605323461142787, 32318605747689071742310867751273886645])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([317408271159096505221916390262196469787, 49717575675851229633286204770028665263])).unwrap(), Fq::new(U256([230095008714744652782426759714312292445, 34768179553549400629211217553610554586])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([259609023305192462223688164733847327699, 55498873698881710203435569648274404478])).unwrap(), Fq::new(U256([256325550082466559964506871761831536884, 46357653853657184428683736043713888406])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([174577378801081039438668831531067795857, 53157547858752109585449689851969780390])).unwrap(), Fq::new(U256([82151408345597494862089436182070514285, 16466308473185906030465208465158400278])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([245839213042247079917412337301921026922, 22457019238909031929740892448482221879])).unwrap(), Fq::new(U256([109513130235370254599453293800543710331, 47936140417415756744920707247895308699])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([10776161146456696103487000724300150234, 21638563281306406703950273508675616623])).unwrap(), Fq::new(U256([207085916216710302318982426148546767798, 7534717897684081051796825732586695012])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([155690435801319682239901337205468120840, 57696234800836220244083348626584391674])).unwrap(), Fq::new(U256([126757665472044690642504190674454727065, 27689844381974512633117281405461511493])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([111371524430469273469505773333501881358, 41915381107432591576214180621073813739])).unwrap(), Fq::new(U256([8591307386286176660318558848651389629, 19322060932256059989553597776338707006])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([34843642275668259465089406915935512474, 25199006933559492770352887141455099490])).unwrap(), Fq::new(U256([10645844381812821922686446509476566858, 35688555865712593739042131731320400759])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([189947043026848705532544415056376972054, 16798314122488336019972329871173393651])).unwrap(), Fq::new(U256([308591971026665862143792543801958855960, 31580607213628921122716343407923691193])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([163093806147198705684115256712150030567, 61044717475711501197927838344963107641])).unwrap(), Fq::new(U256([307478741843426518207523544103444625011, 32813498165003357356133773951397928550])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([95291548361985054080716711033104720817, 32740343556226865497613504752972186774])).unwrap(), Fq::new(U256([99656074525833092595857684423204299816, 52556239373424263650635019541397500598])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([131443621763425639580622258813874957855, 17694455882904396881053896673232601550])).unwrap(), Fq::new(U256([57787363176533727883752817750279248594, 1322856562657561354884935596241784834])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([41788280636884607828680584839495219344, 51762903848994175839707598487572362464])).unwrap(), Fq::new(U256([327908654890111464339978492292915303796, 47344360741808824501224988176645616993])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([263873593879772810642440946076120656306, 60210233792451977377881584251511840173])).unwrap(), Fq::new(U256([77405826008973124400288647208797749719, 53019697808811506999139211676029205579])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([254578463202394703187741702356839561011, 21096342408485278014732435848864790919])).unwrap(), Fq::new(U256([230473548197181635567135409745417796610, 22147822186055995413422331229074559071])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([336878015160301245042116518989872047941, 3472301140759760061593082376347546380])).unwrap(), Fq::new(U256([16684778019642037346902548590480489922, 63174559216409847644418505532384053845])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([7942031074055265340884029455787976403, 46511448608441944387405531466341650044])).unwrap(), Fq::new(U256([201615416576767918339868727158350994412, 64255187168173681561527383354144167567])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([243308496406996226443557446925779452296, 45849597498959240205941430238383669623])).unwrap(), Fq::new(U256([48147614977633577164856356428453305437, 24480032415168951986931189038013914221])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([74830884455458736654394458059429787198, 19093844097308332947749289150298249126])).unwrap(), Fq::new(U256([33643899737753431509498018706471379301, 1054576208141392093174173023229894557])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([25010096200987042473393926604632958161, 36666296164531908118513449509789076579])).unwrap(), Fq::new(U256([33471717379111372311065514803349918045, 39155528164590001712332400960607038189])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([130247467451091212947023092388792369109, 5581090585326037371051920053369586280])).unwrap(), Fq::new(U256([322360327312842929444256001314179239601, 39584478520709259399389327249557202594])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([109138162516199992788447200019766811785, 47525457821584643247627414608651841972])).unwrap(), Fq::new(U256([76309968110600274074720854748516338453, 54739068402347163521228676077893779893])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([42180409310074737345249452446852603163, 4374831491626595184699821621506201010])).unwrap(), Fq::new(U256([130057601620482713644106051600317093239, 12010679381934251249551375242240386988])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([203271863519350650976092478448210829293, 52275506975977654460248547356331731574])).unwrap(), Fq::new(U256([87019891938199989232789187109647269328, 44030460834732836505939430240249354505])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([110171920943726136564495503552694726540, 18110786866783187141705660513534498930])).unwrap(), Fq::new(U256([86561692963802460817471572747090778194, 16449770553500671507593674575394564582])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([221876651023252602014468911202182579931, 20555077542463658872780769250687782117])).unwrap(), Fq::new(U256([121263539082468326970413896423548540112, 61907715120666399953423277772754945448])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([290357192776207049084050762947655748975, 57738753410091871200747406135207764151])).unwrap(), Fq::new(U256([135358190114239175182066780052123869069, 56081507988571629614406051210751609147])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([195622544431419990666685034787625747326, 54874770374981417815900652767969731611])).unwrap(), Fq::new(U256([6243075893789260529022348241785972764, 56521196313521047823704039910006760241])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([243495103440406073534289072917982313090, 57589106164518822864160786869003317813])).unwrap(), Fq::new(U256([235848100009165241948327269818774314127, 37511141642844469126769577281189903937])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([78592775623001882471215007061407141105, 34941191764695857355641077076235001298])).unwrap(), Fq::new(U256([267356444067983260493350102628306009780, 28391440678450825886400816194304466425])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([315751120363918319477358949808863604966, 2531298718421943078485957285990819438])).unwrap(), Fq::new(U256([242692025957487623878146464744369629015, 52949313651228011397330521709355671746])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([223701370913329345484604486557612973051, 34658421329859066183602046886271845301])).unwrap(), Fq::new(U256([239281725801587867757289553693045231533, 14055606950515704853984812857124180593])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([321292144065398009067550865125623060600, 18828918489183113618746569726487918435])).unwrap(), Fq::new(U256([44953266506250560322305758112518632685, 12818035770624926325618902943072758802])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([198712001500603483304766131655796332238, 37755493054877351200992330081944773222])).unwrap(), Fq::new(U256([203494901390506520787336076745404879355, 9719695048988798612029815523720059188])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([324082713664478232334150476671698670935, 57968755532718074766231332303127747698])).unwrap(), Fq::new(U256([172967914786650717580471091827836918517, 39836011047291872323399953583350854813])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([104693003047177326696685168159983773290, 21362453846755563999603040010467525220])).unwrap(), Fq::new(U256([56374029098225126238873495696162227687, 61374137561202742141169238350854689065])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([158941255788517659316631775720407312680, 2477940982651390357755438379586767771])).unwrap(), Fq::new(U256([212374631437158170731256215648153937637, 51786402687200191668238627015343248084])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([67431426417348416607065438525684723676, 59371917376780773574999923335758781547])).unwrap(), Fq::new(U256([329921395426288970367695061271959172005, 18586382421107856878975478269722973139])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([207312856868420470351121539398556490708, 6749469271717320482346458817170377645])).unwrap(), Fq::new(U256([297463942398989836675066926378547228117, 42180709737629195771943741993025816145])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([294946811578231549301710677543341218799, 20992621396953319301177509266206083412])).unwrap(), Fq::new(U256([334489506001987366917405458471890123881, 5644882162624043756459843063788196004])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([339439573540481749394421794993912483199, 45246606008318294125455064627263238236])).unwrap(), Fq::new(U256([270454941633781906426514756020454116063, 13672349860189847846584888881208438638])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([184109810967303467375292693011959132443, 22047527326459095654736206729881706277])).unwrap(), Fq::new(U256([306855976122543936818762044270256507301, 13817693020515802987708208420701920996])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([18944943876649574380290951381571111173, 54681599103691370082120691137707858843])).unwrap(), Fq::new(U256([274481158071926596330090943889570839876, 27656594457831811691825162550052997369])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([72068179718393541085353916977402777797, 23085120169523399820600846468009840798])).unwrap(), Fq::new(U256([39623239893391127442673085065505018616, 49714238177847283522861975755429582165])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([326109292257788796875387755492793672492, 38017828106840559651298055936334805586])).unwrap(), Fq::new(U256([300970874956733001966255469483871058465, 61693366793354972449623979259561628170])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([247117608403708963869106314083852845952, 6270392892469923670893298889464309224])).unwrap(), Fq::new(U256([85229127169148541816114014402805254942, 23692121466582475015916841187427174063])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([322377160078698123442118813943355848949, 44790955831224275605828973863316334451])).unwrap(), Fq::new(U256([246121457999548771204074791900055840778, 36237050391238930724599650306620108395])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([6479203335512729615298164829455402801, 18899857059703106109444283718983930606])).unwrap(), Fq::new(U256([307909459343553081188134777979054198524, 23104930848877581327335450138633822165])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([117408687141894236273811591312978416850, 27064268635283187986411799061361560149])).unwrap(), Fq::new(U256([89156958054425667145216654832344296184, 28498155840544447742658664925302722255])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([264148561516468171084137463578412456470, 56169366101528802291286278394614323282])).unwrap(), Fq::new(U256([80748178838045963455037482306103002135, 31627837956246452962075717178352705444])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([228462396223758741175275124462032775241, 541217865154375519097785562072294622])).unwrap(), Fq::new(U256([147690209661782719555035340726175029695, 47332569057322758675859735205860987883])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([332785070356059610611838282826287767824, 28411804506527712891888004500289036321])).unwrap(), Fq::new(U256([80347503246503526995863756906237840538, 40694963808412110626720215103350226072])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([73246772632733278293170183313248743910, 51739954405360462951359314934075204478])).unwrap(), Fq::new(U256([323288654986485051471927026636004277374, 55328786519754064217639655113348868700])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([107183915141555030394480811055835932411, 28743337205734819613101267731330261274])).unwrap(), Fq::new(U256([265627562010734120334801493421007962860, 21113833505065027987742094370311235794])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([145390576010968264278778301175950417032, 25603093564950634850451680305056487937])).unwrap(), Fq::new(U256([171224041010792018070371064846900704811, 50628689539088695975060303147297926726])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([181636860313080877557100227907605412643, 19106550849436726002022081116470620327])).unwrap(), Fq::new(U256([97037887458346176230370696042525047476, 38315199704917257726469887714210863756])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([165755565692898412861884552154775259871, 23293168176203049828541858307098176114])).unwrap(), Fq::new(U256([111579679857180892572143097811395790681, 21526844626039124934190021704344618627])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([335258764315672309726009417334449368453, 103404408330729181499563204707141002])).unwrap(), Fq::new(U256([191071126894978514432059671553769626880, 5078535341242500657674641815850178322])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([72579254643315478022492189073789591450, 44167090301255006150406576419615402343])).unwrap(), Fq::new(U256([183377394594390630206602640102963629039, 23990882412966103396841140729654425350])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([66679892701097417538832992488778691458, 58183623071000462877229489086029090922])).unwrap(), Fq::new(U256([294287188350011499287443971281382798030, 18151493196929461249661140690411179165])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([74805142807574287323605988196524192535, 18100483139591925759594156937304667397])).unwrap(), Fq::new(U256([139120256621259665435015294709846943744, 33412960622211929574259219789625651855])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([220224993980394484792559335583346867939, 59970476024877867633551960552156066815])).unwrap(), Fq::new(U256([165016476912320413641798011850430383400, 40087637389842239140931509158661317913])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([26669149284356063486638395246213038984, 48932308059862148467989673144064276455])).unwrap(), Fq::new(U256([112017260916309235481572722060511554523, 1292432061742789600481111238179731344])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([162967204317649679020730851544536806310, 32746563765578881184210706548921981702])).unwrap(), Fq::new(U256([337655071624981772776297963145482993016, 33145701869840355692613219913227327224])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([116821007696021810040981824168902376385, 46826353416371587869648603174437608629])).unwrap(), Fq::new(U256([286148963257224477643539266261574758187, 20903220425259595580283778435979469601])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([265822762895337622674318289864003560145, 62593533822281897460585212079268870737])).unwrap(), Fq::new(U256([180099245396770402139578639916315358651, 5289891249371566111607452004558281256])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([215101732208902323596827687492609836402, 60151379040875297433910120987825047988])).unwrap(), Fq::new(U256([200376206977547885622799626268846642009, 36107786828287031009478318323095814795])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([228632312969356396843036040954049410317, 29609317291386275502321244693464627541])).unwrap(), Fq::new(U256([250663071554126685512943663183005740350, 26707628083885320745488595167117920892])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([167686658731994607869572641962208584698, 32951815034866853047881968952398296677])).unwrap(), Fq::new(U256([59732843077777643290110085014121270991, 19690792480201713304572701985281652804])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([231501326469976058677041937508913323742, 58052555179693233540250119022219467367])).unwrap(), Fq::new(U256([268839178939407644460103390562050250791, 51244123519915889574699519125947948433])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([15335721378364717505397406493265760457, 50543654003772338617069451450467156120])).unwrap(), Fq::new(U256([147668560194510325215853847922114742688, 63436405408432434981818242494869996954])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([324251215335356818847634371978327094047, 53597895382917031159515977918983297046])).unwrap(), Fq::new(U256([336456284126450597413571076848285070208, 9834107003348358663468869495065030828])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([4893147365574829346121011245181836636, 48500338802265501645481354399443896938])).unwrap(), Fq::new(U256([242843203647079174857073639694991742532, 39587639177217829196831837775949647839])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([260649755517395052354543412976006056345, 41595453161552128661866319661777368912])).unwrap(), Fq::new(U256([186473459113497941493205881377276393540, 10232971634855792150439771169030969984])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([153156583597535672033657311231041916285, 12870713391441384499488044032601826356])).unwrap(), Fq::new(U256([160011046903032247780660416127755862604, 59740658966179967701258870119449151446])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([53781592683706300755113958689829175735, 5160246898475719650527746461036900035])).unwrap(), Fq::new(U256([127406077100448076503048857464807666048, 5027357614298608514576108334690214373])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([123323183388079598137911866134337668968, 30831375761327507469744048674597209517])).unwrap(), Fq::new(U256([282757148554018853521114389661374143994, 41725892078174834143829265912887863334])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([231142251602796484920728141361531696801, 43369687276260799996121293373394088506])).unwrap(), Fq::new(U256([193386979673076292922455602081629373614, 13517077812823754339540367661413555482])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([32945933209167472962271089890685376310, 23645956766949617286909797728539496219])).unwrap(), Fq::new(U256([128873344219688070096765758438736759675, 54069131826062143440208582926638727688])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([307107650784744827303698892877513591771, 60627460508970098579939906160617347629])).unwrap(), Fq::new(U256([12937320851881845879608263929438137281, 2854427512588805421842219517935335360])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([262645642925640470181426114122199935672, 23413555848620303130105885294683582394])).unwrap(), Fq::new(U256([23944562046020585348174388998952320194, 45084472027505841447432550475379032518])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([270949855981998457946423634423510659239, 52707980377915683827433393510154134085])).unwrap(), Fq::new(U256([171166926104504281831828631675145047352, 56745888676097625411354427758332611365])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([185108543406354812098225836322332681649, 38838551534016041901193249287881822376])).unwrap(), Fq::new(U256([220861944829798693206584739921489268335, 13318922669002030127185287983351463417])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([262922874647306006405720155830997014046, 60319938491997842161822791958118330349])).unwrap(), Fq::new(U256([267514688756785010198929321975858316586, 13431251784763107399729556927577885114])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([305051645126359797430782278231595423685, 47480212315012773203275805598650271426])).unwrap(), Fq::new(U256([61134576323692903306240935503639110628, 44357224521089864059608299955085658945])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([288996190497897556231840221598596395445, 7566958477031738465971309099100507442])).unwrap(), Fq::new(U256([133814060971580680876860604680251720534, 59538638355154696335532006425868258220])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([255893828824714285725760712618255298338, 62054226651551009305279711173141456137])).unwrap(), Fq::new(U256([278488875745654026562827558863613836490, 21008509582420141439141693540430274581])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([275369357162135051507223962619748317962, 42306683939383379782070254815636483252])).unwrap(), Fq::new(U256([33692084190064829046021773249716169044, 16341905693988596015816579527114077079])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([50744916563901235043837175537752247974, 10172930551818555335663216915059687204])).unwrap(), Fq::new(U256([130434124352272975884189586048532260990, 23515228462564504103971218058319726320])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([315580665939248091007437797584737352150, 29708290546079112870849940807626622733])).unwrap(), Fq::new(U256([74442409874403535705044740324158268395, 1505199293980260993350511103843027993])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([250928935731107472086814134230515792409, 5910959187805062997477270179924851551])).unwrap(), Fq::new(U256([280398933657384735588696583124789483429, 54783439113329460653893087793006061735])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([251724005393716093632013371894721049773, 58848512269437108583523559065211202801])).unwrap(), Fq::new(U256([31822195116686689031909059702809975955, 24789755710735999475756589726034740758])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([311444467767445140793669110935330311306, 49680899512402416135008064018078765713])).unwrap(), Fq::new(U256([260075513883026034919987130520464438138, 71886459212490540714376620236740887])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([218811104197169142519581806122315915406, 2003975639040616906277810278389252752])).unwrap(), Fq::new(U256([162449781759514417953768678035934414521, 20826616094394726852079185751274539294])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([50122791277121629311774878848574398797, 41715208027917549489096484827508261925])).unwrap(), Fq::new(U256([61219715357759165032395798940008313361, 55168729445645977375765036089053114792])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([79415664581913745231517567945561341244, 19961313692291707660858657690356279927])).unwrap(), Fq::new(U256([169697929611525566517558862344260578764, 32844991405556424161021671060175948696])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([97951068758528710273694591785781637499, 18749313896215455583118043227454839626])).unwrap(), Fq::new(U256([86416223342992670848217929911878122430, 45508713650090104780284816480988042321])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([205166366708889542887913296220698256110, 64283667930878381361054417761490454821])).unwrap(), Fq::new(U256([127337780289640176933683040648197123026, 45189848010560984741741084451460237213])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([339284773562100413239152438122508204497, 50631498218339719105248005011418978450])).unwrap(), Fq::new(U256([33417073692493560040015373020490319634, 35461627756915585142929292352046627148])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([65706590332479893819786959846945487398, 49780412881756049082474794087182699607])).unwrap(), Fq::new(U256([52838637386752308153448355705238086197, 25418435995504719731730063829115376943])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([272342513391057040499052435082268704444, 40734611488359738470851428580967866388])).unwrap(), Fq::new(U256([195464905670800875597868841505577455579, 4567343913211817054294335942572391310])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([116585277318795490267488822790585572050, 60535270002127930300661104257194060095])).unwrap(), Fq::new(U256([60938698090087736292473391678184666708, 13034546647626058553077909902189865751])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([173096860849358760769153772521753734944, 46051492674049147015395144791836438516])).unwrap(), Fq::new(U256([217847244022302146556080020387033222160, 3714600728561149832835084325311638745])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([86749998713441153819371623717671780463, 22072421598347798931100735874660737662])).unwrap(), Fq::new(U256([318879363093875678699747171034610586590, 20824335590094416898946169150944599166])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([64488889092438538888826783837043953900, 37184887689753051095864638902885179863])).unwrap(), Fq::new(U256([30545663928156310707528798028105042841, 35594636889744346303301050561815356233])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([25983064025798529721395241242693795182, 8818840193887973907465639723423785685])).unwrap(), Fq::new(U256([258083038864362618956338229292791076964, 59924965332411444209484507096908691451])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([220038260912754873591576914680586914279, 13066051346629987308799710602199271472])).unwrap(), Fq::new(U256([145490690982645862542497983334649610263, 61905193484912278316593689230425266660])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([91140715068468329987534450751429662322, 19286363177619277557153390898202185703])).unwrap(), Fq::new(U256([93820352751188715792914230189639637744, 34108640942617057347076989500332659863])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([64028159517497609933876978072339213461, 15242545675596643010631683990284567829])).unwrap(), Fq::new(U256([196818140250582584748747602707707107768, 19674391967426090834762688793787807910])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([240272916169613868056785616677104041776, 43506134861603450917738367078393340621])).unwrap(), Fq::new(U256([4195648020339163006604321850508278540, 4672904647050343041579830671683109182])).unwrap()) },
            EllCoeffs { ell_0: Fq2::new(Fq::new(U256([76224289879065773445423898190273853011, 29552147447995384759425668183293749353])).unwrap(), Fq::new(U256([131775855227043960655312514543203001277, 11525949759137095338012468817574456927])).unwrap()), ell_vw: Fq2::new(Fq::new(U256([24164999216876611682317527533876599098, 1468136824339634997687523096410027135])).unwrap(), Fq::new(U256([287955683679784091578386069448645460818, 42392463637589581183769087503942163916])).unwrap()), ell_vv: Fq2::new(Fq::new(U256([277415168387483520146499744266583303231, 43682907322954483721445867111493680565])).unwrap(), Fq::new(U256([231497467135626786731515493713118585761, 56904388641135605566397108514408579757])).unwrap()) }
        ]
    };

    assert!(expected_g2_p == g2_p);
    assert!(expected_g2_p.coeffs.len() == 87);
}

pub fn pairing(p: &G1, q: &G2) -> Fq12 {
    match (p.to_affine(), q.to_affine()) {
        (None, _) | (_, None) => Fq12::one(),
        (Some(p), Some(q)) => q.precompute()
            .miller_loop(&p)
            .final_exponentiation()
            .expect("miller loop cannot produce zero"),
    }
}

pub fn pairing_batch(ps: &[G1], qs: &[G2]) -> Fq12 {
    let mut p_affines: Vec<AffineG<G1Params>> = Vec::new();
    let mut q_precomputes: Vec<G2Precomp> = Vec::new();
    for (p, q) in ps.into_iter().zip(qs.into_iter()) {

        let p_affine = p.to_affine();
        let q_affine = q.to_affine();
        let exists = match(p_affine, q_affine)
        {
            (None, _) | (_, None) => false,
            (Some(_p_affine), Some(_q_affine)) => true,
        };

        if exists {
            p_affines.push(p.to_affine().unwrap());
            q_precomputes.push(q.to_affine().unwrap().precompute());    
        }
    }
    if q_precomputes.len() == 0 {
        return Fq12::one();
    }
    miller_loop_batch(&q_precomputes, &p_affines).final_exponentiation().expect("miller loop cannot produce zero")
}

#[test]
fn test_reduced_pairing() {
    use fields::Fq6;

    let g1 = G1::one()
        * Fr::from_str(
            "18097487326282793650237947474982649264364522469319914492172746413872781676",
        ).unwrap();
    let g2 = G2::one()
        * Fr::from_str(
            "20390255904278144451778773028944684152769293537511418234311120800877067946",
        ).unwrap();

    let gt = pairing(&g1, &g2);

    let expected = Fq12::new(
        Fq6::new(
            Fq2::new(
                Fq::from_str(
                    "7520311483001723614143802378045727372643587653754534704390832890681688842501",
                ).unwrap(),
                Fq::from_str(
                    "20265650864814324826731498061022229653175757397078253377158157137251452249882",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "11942254371042183455193243679791334797733902728447312943687767053513298221130",
                ).unwrap(),
                Fq::from_str(
                    "759657045325139626991751731924144629256296901790485373000297868065176843620",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "16045761475400271697821392803010234478356356448940805056528536884493606035236",
                ).unwrap(),
                Fq::from_str(
                    "4715626119252431692316067698189337228571577552724976915822652894333558784086",
                ).unwrap(),
            ),
        ),
        Fq6::new(
            Fq2::new(
                Fq::from_str(
                    "14901948363362882981706797068611719724999331551064314004234728272909570402962",
                ).unwrap(),
                Fq::from_str(
                    "11093203747077241090565767003969726435272313921345853819385060670210834379103",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "17897835398184801202802503586172351707502775171934235751219763553166796820753",
                ).unwrap(),
                Fq::from_str(
                    "1344517825169318161285758374052722008806261739116142912817807653057880346554",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "11123896897251094532909582772961906225000817992624500900708432321664085800838",
                ).unwrap(),
                Fq::from_str(
                    "17453370448280081813275586256976217762629631160552329276585874071364454854650",
                ).unwrap(),
            ),
        ),
    );

    assert_eq!(expected, gt);
}

#[test]
fn predefined_pair() {
    let g1 = AffineG1::new(
        Fq::from_str("1").expect("Fq(1) should exist"),
        Fq::from_str("2").expect("Fq(2) should exist"),
    ).expect("Point (1,2) should exist in G1")
        .to_jacobian();

    let g2 = AffineG2::new(
        Fq2::new(
            Fq::from_str("10857046999023057135944570762232829481370756359578518086990519993285655852781")
                .expect("a-coeff of g2 x generator is of the right order"),
            Fq::from_str("11559732032986387107991004021392285783925812861821192530917403151452391805634")
                .expect("b-coeff of g2 x generator is of the right order"),
        ),
        Fq2::new(
            Fq::from_str("8495653923123431417604973247489272438418190587263600148770280649306958101930")
                .expect("a-coeff of g2 y generator is of the right order"),
            Fq::from_str("4082367875863433681332203403145435568316851327593401208105741076214120093531")
                .expect("b-coeff of g2 y generator is of the right order"),
        ),
    ).expect("Point(11559732032986387107991004021392285783925812861821192530917403151452391805634 * i + 10857046999023057135944570762232829481370756359578518086990519993285655852781, 4082367875863433681332203403145435568316851327593401208105741076214120093531 * i + 8495653923123431417604973247489272438418190587263600148770280649306958101930) is a valid generator for G2")
        .to_jacobian();

    let p = pairing(&g1, &g2);

    let g1_vec : Vec<G1> = vec![g1, g1];
    let g2_vec : Vec<G2> = vec![g2, g2];
    let p2 = pairing_batch(&g1_vec, &g2_vec);
    assert!(!p2.is_zero());
    assert!(!p.is_zero());
}

#[test]
fn test_batch_bilinearity_empty() {
    let p_vec : Vec<G1> = Vec::new();
    let q_vec : Vec<G2> = Vec::new();
    let r = pairing_batch(&p_vec, &q_vec);
    assert_eq!(r, Fq12::one());
}

#[test]
fn test_batch_bilinearity_one() {
    use rand::{SeedableRng, StdRng};
    let seed = [
        0, 0, 0, 0, 0, 0, 64, 13, // 103245
        0, 0, 0, 0, 0, 0, 176, 2, // 191922
        0, 0, 0, 0, 0, 0, 0, 13, // 1293
        0, 0, 0, 0, 0, 0, 96, 7u8, // 192103
    ];
    let mut rng = StdRng::from_seed(seed);
    let p_vec : Vec<G1> = vec![G1::random(&mut rng)];
    let q_vec : Vec<G2> = vec![G2::random(&mut rng)];
    let s = Fr::random(&mut rng);
    let sp_vec : Vec<G1> = vec![p_vec[0] * s];
    let sq_vec : Vec<G2> = vec![q_vec[0] * s];
    let b = pairing_batch(&sp_vec, &q_vec);
    let c = pairing_batch(&p_vec, &sq_vec);
    assert_eq!(b, c);
}

#[test]
fn test_batch_bilinearity_fifty() {
    use rand::{SeedableRng, StdRng};
    let seed = [
        0, 0, 0, 0, 0, 0, 64, 13, // 103245
        0, 0, 0, 0, 0, 0, 176, 2, // 191922
        0, 0, 0, 0, 0, 0, 0, 13, // 1293
        0, 0, 0, 0, 0, 0, 96, 7u8, // 192103
    ];
    let mut rng = StdRng::from_seed(seed);

    let mut p_vec : Vec<G1> = Vec::new();
    let mut q_vec : Vec<G2> = Vec::new();
    let mut sp_vec : Vec<G1> = Vec::new();
    let mut sq_vec : Vec<G2> = Vec::new();
    
    for _ in 0..50 {
        let p = G1::random(&mut rng);
        let q = G2::random(&mut rng);
        let s = Fr::random(&mut rng);
        let sp = p * s;
        let sq = q * s;
        sp_vec.push(sp);
        q_vec.push(q);
        sq_vec.push(sq);
        p_vec.push(p);
    }
    let b_batch = pairing_batch(&sp_vec, &q_vec);
    let c_batch = pairing_batch(&p_vec, &sq_vec);
    assert_eq!(b_batch, c_batch);
}

#[test]
fn test_bilinearity() {
    use rand::{SeedableRng, StdRng};
    let seed = [
        0, 0, 0, 0, 0, 0, 64, 13, // 103245
        0, 0, 0, 0, 0, 0, 176, 2, // 191922
        0, 0, 0, 0, 0, 0, 0, 13, // 1293
        0, 0, 0, 0, 0, 0, 96, 7u8, // 192103
    ];
    let mut rng = StdRng::from_seed(seed);

    for _ in 0..50 {
        let p = G1::random(&mut rng);
        let q = G2::random(&mut rng);
        let s = Fr::random(&mut rng);
        let sp = p * s;
        let sq = q * s;

        let a = pairing(&p, &q).pow(s);
        let b = pairing(&sp, &q);
        let c = pairing(&p, &sq);

        assert_eq!(a, b);
        assert_eq!(b, c);

        let t = -Fr::one();

        assert!(a != Fq12::one());
        assert_eq!((a.pow(t)) * a, Fq12::one());
    }
}

#[test]
fn internals() {
    let test_p = G1::one();

    let val = G1::new(test_p.x().clone(), test_p.y().clone(), test_p.z().clone());

    let affine = val.to_affine()
        .expect("There should be affine coords for (0, 0)");

    assert_eq!(affine.x(), &Fq::one());
}

#[test]
fn affine_fail() {
    let res = AffineG1::new(Fq::one(), Fq::one());
    assert!(
        res.is_err(),
        "Affine initialization should fail because the point is not on curve"
    );
}

#[test]
fn affine_ok() {
    let res = AffineG1::new(Fq::one(), G1Params::coeff_b());
    assert!(
        res.is_err(),
        "Affine initialization should be ok because the point is on the curve"
    );
}

#[test]
fn test_y_at_point_at_infinity() {
    assert!(G1::zero().y == Fq::one());
    assert!((-G1::zero()).y == Fq::one());

    assert!(G2::zero().y == Fq2::one());
    assert!((-G2::zero()).y == Fq2::one());
}
