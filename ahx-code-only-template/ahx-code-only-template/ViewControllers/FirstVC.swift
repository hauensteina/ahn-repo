//
//  FirstVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-17.
//

import UIKit

//=================================
class FirstVC: UIViewController {
    var btnSecond:UIButton!
    
    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        let w = UIScreen.main.bounds.width
        let h = UIScreen.main.bounds.height
        btnSecond = UIButton( type: .system,
                              primaryAction: UIAction(title: "Go to Second", handler: { _ in
                                AHXMain.shared.topVC( "SecondVC")
        }))
        var v = btnSecond!
        AHXLayout.width( v, x: 0.33 * w)
        AHXLayout.height( v, y: AHXConstants.btnHeight)
        AHXLayout.center( v, x: w/2)
        AHXLayout.middle( v, y: h/2)
        self.view.addSubview( v)
    } // viewDidLoad()
} // class FirstVC

