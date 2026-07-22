# Rotating projected CF parameters

OK, lets start again with simpler numbers: 
paramA = {"C21": +100+100j, "C22":  -100j,}       
paramB = {"C21": +100-100j, "C22":  +100j,} 
These are related only by a complex conjugation. 
The rotation 
alpha, beta, gamma = 3.9269908169872414, 0.9553166181245092, 0.0
is 
alpha = np.pi / 4 + np.pi 
beta = np.arccos(1/np.sqrt(3))
gamma = 0.0


This should rotate a cube with xyz though the faces, so that the 111 corner is along z. 


I’m solving the phase relations directly from the coefficients, then I’ll test the candidate inverse rotations.
Yes — **both can be rotated to pure `C20`**, but with **different `alpha`**.

For this helper, use the inverse Euler order:

```python
rotate_cf_params(coeff, -gamma, -beta, -alpha)
```

For your two sets:

- `paramB = {"C21": 100-100j, "C22": 100j}`  
  corresponds to  
  `alpha = pi/4`, `beta = arccos(1/sqrt(3))`, `gamma = 0`

- `paramA = {"C21": 100+100j, "C22": -100j}`  
  corresponds to  
  `alpha = 7*pi/4` (`-pi/4`), `beta = arccos(1/sqrt(3))`, `gamma = 0`

So the calls that give only `C20` are:

```python
beta = np.arccos(1/np.sqrt(3))

rotate_cf_params(paramB, 0.0, -beta, -np.pi/4)
rotate_cf_params(paramA, 0.0, -beta, -7*np.pi/4)
```

Both return essentially only `C20` nonzero.

For the two `C20`-only rotations:

- **paramB** → **`(+1, +1, +1)`** corner, i.e. the `[111]` body diagonal
- **paramA** → **`(+1, -1, +1)`** corner, i.e. the `[1\bar{1}1]` body diagonal

So they’re different three-fold axes related by a 90° rotation about `z`.

It seems that we need to complex conjugate the parameters projected from molcas and translated from Stevens to Wybourne normalization. 

You’re likely right, and the reason is visible in  read_cf.py :                      ┃
                                                                                    
   •  stevens_to_wybourne_chilton  builds positive- q  terms as                         ┃
      C_{kq} = λ (B_{kq} + i B_{k,-q})  ( read_cf.py:178 )                              ┃
   • Then it enforces                                                                   ┃
      C_{k,-q} = (-1)^q * conj(C_{kq})  ( read_cf.py:180 )                              ┃
                                                                                    
   That is a valid Wybourne convention, but if your downstream ( pycf ) expects the     
   opposite azimuthal phase convention, it wants                                        
                                                                                    
   •  C'_{kq} = λ (B_{kq} - i B_{k,-q}) = conj(C_{kq})                                                 
   So your observation (“it behaves correctly only after conjugating Wybourne params”) is exactly what you’d expect from a convention mismatch, not a random bug.           
                                                                                    
   The practical spot to apply this is where  chilton_pycf  is created                 ( read_cf.py:403-405 ): conjugate those  q>=0  values before emitting/using them for pycf.
                                                                                    
 ❯ What is the origin of the convention diferences?                               
 
  ┃They come from **equally valid but different definition choices** in spherical-tensor algebra.

Main sources:

1. **Phase of spherical harmonics / tensors**  
   Some use \(e^{+iq\phi}\), others \(e^{-iq\phi}\).  
   Switching this effectively conjugates complex \(C_{kq}\)-type coefficients.

2. **Operator vs basis convention**  
   Whether you define \(T_{kq}\) or \(T_{kq}^\dagger\) as the named object changes where complex conjugation lands.

3. **Active vs passive rotations**  
   Rotating states/operators vs rotating coordinate axes uses inverse rotations; same physics, different angle usage/signs.

4. **Index/sign placement in \(q\leftrightarrow -q\) relations**  
   Different texts place \((-1)^q\) and conjugation in equivalent but differently arranged formulas.

5. **Historical community conventions**  
   Stevens/extended-Stevens, Wybourne, spectroscopic-code conventions (Molcas, McPhase, etc.) were developed in different subfields and not fully unified.

So what you’re seeing is normal: two pipelines can both be correct internally, yet differ by conjugation/inverse-angle mapping when connected.
          
