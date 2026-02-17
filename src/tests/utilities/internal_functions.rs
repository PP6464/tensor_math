#[cfg(test)]
mod internal_functions_tests {
    use float_cmp::approx_eq;
    use num::complex::Complex64;
    use crate::utilities::internal_functions::{bluestein_fft_vec, fft_vec, ifft_vec, radix_2_fft_vec};

    #[test]
    fn radix_2_vec_fft_test() {
        let v1 = vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 2.0), Complex64::new(4.0, -5.0),
            Complex64::new(6.0, 7.0), Complex64::new(8.0, -4.0),
            Complex64::new(2.0, 1.0), Complex64::new(0.0, 0.0),
        ];
        let v2 = vec![
            Complex64 { re: 0.0, im: 1.0 },
            Complex64 { re: -0.1453217681275245, im: -3.6026214877097886 },
            Complex64 { re: -1.3968022466674206, im: 0.2212317420824741 },
            Complex64 { re: 6.39622598104463, im: -0.29714171603742257 },
            Complex64 { re: -2.3576391889522714, im: 10.365400978964416 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
        ];
        let res1 = radix_2_fft_vec(&v1);
        let res2 = radix_2_fft_vec(&v2);

        let ans1 = vec![
            Complex64 { re: 26.0, im: 0.0 },
            Complex64 { re: -12.485281374238571, im: -0.9289321881345245 },
            Complex64 { re: 2.0, im: -2.0000000000000018 },
            Complex64 { re: -0.34314575050761853, im: -10.242640687119287 },
            Complex64 { re: -2.0, im: 20.0 },
            Complex64 { re: 4.485281374238571, im: -15.071067811865476 },
            Complex64 { re: 2.0, im: 10.000000000000002 },
            Complex64 { re: -11.656854249492381, im: -1.7573593128807126 },
        ];
        let ans2 = vec![
            Complex64 { re: 2.4964627772974133, im: 7.686869517299678 },
            Complex64 { re: 10.194430302685799, im: -4.794067509264503 },
            Complex64 { re: -4.804250848251911, im: -14.725982651422731 },
            Complex64 { re: -18.400909719645895, im: 0.9514637996861768 },
            Complex64 { re: -4.266316713957217, im: 17.685716986054096 },
            Complex64 { re: 13.946932312659404, im: 5.899646309853949 },
            Complex64 { re: 4.004431213373595, im: -12.84493282150859 },
            Complex64 { re: -15.4761803961386, im: -4.600523633014343 },
            Complex64 { re: -10.005345648536798, im: 15.4863959247941 },
            Complex64 { re: 8.873863904133222, im: 13.797591498439782 },
            Complex64 { re: 9.961992710321402, im: -1.211214813171261 },
            Complex64 { re: -0.041646627012200454, im: -2.0042344264809056 },
            Complex64 { re: 2.3446428293875154, im: 4.602621487709789 },
            Complex64 { re: 8.446377396379242, im: -1.472613543220143 },
            Complex64 { re: 0.2683836803659996, im: -8.679473629755085 },
            Complex64 { re: -7.542867173060969, im: 0.2227375039999857 },
        ];

        for i in 0..8 {
            assert!(approx_eq!(f64, res1[i].re, ans1[i].re, epsilon = 1e-15));
            assert!(approx_eq!(f64, res1[i].im, ans1[i].im, epsilon = 1e-15));
        }

        for i in 0..16 {
            assert!(approx_eq!(f64, res2[i].re, ans2[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res2[i].im, ans2[i].im, epsilon = 1e-10));
        }
    }

    #[test]
    fn bluestein_fft_test() {
        let v1 = vec![
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, -3.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(5.0, -4.0),
            Complex64::new(8.0, -7.0)
        ];
        let res = bluestein_fft_vec(&v1);
        let ans = vec![
            Complex64 { re: 16.0, im: - 14.0},
            Complex64 { re: 3.8036497995578236, im: 10.012395135066075 },
            Complex64 { re: -6.738096517215357, im:  7.267570420448962 },
            Complex64 { re: -5.734039437784222, im: 7.822599523300513 },
            Complex64 { re: -7.331513844558244, im: -6.102565078815551 },
        ];

        for i in 0..5 {
            assert!(approx_eq!(f64, res[i].re, ans[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res[i].im, ans[i].im, epsilon = 1e-10));
        }
    }

    #[test]
    fn fft_test() {
        let v1 = vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 2.0), Complex64::new(4.0, -5.0),
            Complex64::new(6.0, 7.0), Complex64::new(8.0, -4.0),
            Complex64::new(2.0, 1.0), Complex64::new(0.0, 0.0),
        ];
        let v2 = vec![
            Complex64 { re: 0.0, im: 1.0 },
            Complex64 { re: -0.1453217681275245, im: -3.6026214877097886 },
            Complex64 { re: -1.3968022466674206, im: 0.2212317420824741 },
            Complex64 { re: 6.39622598104463, im: -0.29714171603742257 },
            Complex64 { re: -2.3576391889522714, im: 10.365400978964416 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
        ];
        let v3 = vec![
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, -3.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(5.0, -4.0),
            Complex64::new(8.0, -7.0),
        ];

        let res1 = fft_vec(&v1);
        let res2 = fft_vec(&v2);
        let res3 = fft_vec(&v3);

        let ans1 = vec![
            Complex64 { re: 26.0, im: 0.0 },
            Complex64 { re: -12.485281374238571, im: -0.9289321881345245 },
            Complex64 { re: 2.0, im: -2.0000000000000018 },
            Complex64 { re: -0.34314575050761853, im: -10.242640687119287 },
            Complex64 { re: -2.0, im: 20.0 },
            Complex64 { re: 4.485281374238571, im: -15.071067811865476 },
            Complex64 { re: 2.0, im: 10.000000000000002 },
            Complex64 { re: -11.656854249492381, im: -1.7573593128807126 },
        ];
        let ans2 = vec![
            Complex64 { re: 2.4964627772974133, im: 7.686869517299678 },
            Complex64 { re: 10.194430302685799, im: -4.794067509264503 },
            Complex64 { re: -4.804250848251911, im: -14.725982651422731 },
            Complex64 { re: -18.400909719645895, im: 0.9514637996861768 },
            Complex64 { re: -4.266316713957217, im: 17.685716986054096 },
            Complex64 { re: 13.946932312659404, im: 5.899646309853949 },
            Complex64 { re: 4.004431213373595, im: -12.84493282150859 },
            Complex64 { re: -15.4761803961386, im: -4.600523633014343 },
            Complex64 { re: -10.005345648536798, im: 15.4863959247941 },
            Complex64 { re: 8.873863904133222, im: 13.797591498439782 },
            Complex64 { re: 9.961992710321402, im: -1.211214813171261 },
            Complex64 { re: -0.041646627012200454, im: -2.0042344264809056 },
            Complex64 { re: 2.3446428293875154, im: 4.602621487709789 },
            Complex64 { re: 8.446377396379242, im: -1.472613543220143 },
            Complex64 { re: 0.2683836803659996, im: -8.679473629755085 },
            Complex64 { re: -7.542867173060969, im: 0.2227375039999857 },
        ];
        let ans3 = vec![
            Complex64 { re: 16.0, im: - 14.0},
            Complex64 { re: 3.8036497995578236, im: 10.012395135066075 },
            Complex64 { re: -6.738096517215357, im:  7.267570420448962 },
            Complex64 { re: -5.734039437784222, im: 7.822599523300513 },
            Complex64 { re: -7.331513844558244, im: -6.102565078815551 },
        ];

        for i in 0..8 {
            assert!(approx_eq!(f64, res1[i].re, ans1[i].re, epsilon = 1e-15));
            assert!(approx_eq!(f64, res1[i].im, ans1[i].im, epsilon = 1e-15));
        }

        for i in 0..16 {
            assert!(approx_eq!(f64, res2[i].re, ans2[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res2[i].im, ans2[i].im, epsilon = 1e-10));
        }

        for i in 0..5 {
            assert!(approx_eq!(f64, res3[i].re, ans3[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res3[i].im, ans3[i].im, epsilon = 1e-10));
        }
    }

    #[test]
    fn ifft_test() {
        let v1 = vec![
            Complex64 { re: 26.0, im: 0.0 },
            Complex64 { re: -12.485281374238571, im: -0.9289321881345245 },
            Complex64 { re: 2.0, im: -2.0000000000000018 },
            Complex64 { re: -0.34314575050761853, im: -10.242640687119287 },
            Complex64 { re: -2.0, im: 20.0 },
            Complex64 { re: 4.485281374238571, im: -15.071067811865476 },
            Complex64 { re: 2.0, im: 10.000000000000002 },
            Complex64 { re: -11.656854249492381, im: -1.7573593128807126 },
        ];
        let v2 = vec![
            Complex64 { re: 2.4964627772974133, im: 7.686869517299678 },
            Complex64 { re: 10.194430302685799, im: -4.794067509264503 },
            Complex64 { re: -4.804250848251911, im: -14.725982651422731 },
            Complex64 { re: -18.400909719645895, im: 0.9514637996861768 },
            Complex64 { re: -4.266316713957217, im: 17.685716986054096 },
            Complex64 { re: 13.946932312659404, im: 5.899646309853949 },
            Complex64 { re: 4.004431213373595, im: -12.84493282150859 },
            Complex64 { re: -15.4761803961386, im: -4.600523633014343 },
            Complex64 { re: -10.005345648536798, im: 15.4863959247941 },
            Complex64 { re: 8.873863904133222, im: 13.797591498439782 },
            Complex64 { re: 9.961992710321402, im: -1.211214813171261 },
            Complex64 { re: -0.041646627012200454, im: -2.0042344264809056 },
            Complex64 { re: 2.3446428293875154, im: 4.602621487709789 },
            Complex64 { re: 8.446377396379242, im: -1.472613543220143 },
            Complex64 { re: 0.2683836803659996, im: -8.679473629755085 },
            Complex64 { re: -7.542867173060969, im: 0.2227375039999857 },
        ];
        let v3 = vec![
            Complex64 { re: 16.0, im: - 14.0},
            Complex64 { re: 3.8036497995578236, im: 10.012395135066075 },
            Complex64 { re: -6.738096517215357, im:  7.267570420448962 },
            Complex64 { re: -5.734039437784222, im: 7.822599523300513 },
            Complex64 { re: -7.331513844558244, im: -6.102565078815551 },
        ];

        let ans1 = vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 2.0), Complex64::new(4.0, -5.0),
            Complex64::new(6.0, 7.0), Complex64::new(8.0, -4.0),
            Complex64::new(2.0, 1.0), Complex64::new(0.0, 0.0),
        ];
        let ans2 = vec![
            Complex64 { re: 0.0, im: 1.0 },
            Complex64 { re: -0.1453217681275245, im: -3.6026214877097886 },
            Complex64 { re: -1.3968022466674206, im: 0.2212317420824741 },
            Complex64 { re: 6.39622598104463, im: -0.29714171603742257 },
            Complex64 { re: -2.3576391889522714, im: 10.365400978964416 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
        ];
        let ans3 = vec![
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, -3.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(5.0, -4.0),
            Complex64::new(8.0, -7.0),
        ];

        let res1 = ifft_vec(&v1);
        let res2 = ifft_vec(&v2);
        let res3 = ifft_vec(&v3);

        for i in 0..8 {
            assert!(approx_eq!(f64, res1[i].re, ans1[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res1[i].im, ans1[i].im, epsilon = 1e-10));
        }

        for i in 0..16 {
            assert!(approx_eq!(f64, res2[i].re, ans2[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res2[i].im, ans2[i].im, epsilon = 1e-10));
        }

        for i in 0..5 {
            assert!(approx_eq!(f64, res3[i].re, ans3[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res3[i].im, ans3[i].im, epsilon = 1e-10));
        }
    }
}