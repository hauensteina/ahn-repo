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
        btnSecond = UIButton( type: .system,
                              primaryAction: UIAction(title: "Go to Second", handler: { _ in
                                AHXMain.shared.topVC( "SecondVC")
                              }))
        self.view.addSubview( btnSecond)
        layout()
    } // viewDidLoad()
    
    //------------------------
    override func layout() {
        let v = btnSecond!
        AHXLayout.width( v, 0.33 * AHC.w)
        AHXLayout.height( v, AHC.btnHeight)
        AHXLayout.center( v, AHC.w / 2)
        AHXLayout.middle( v, AHC.h / 2)
        self.view.addSubview( v)
    } // layout()

} // class FirstVC

