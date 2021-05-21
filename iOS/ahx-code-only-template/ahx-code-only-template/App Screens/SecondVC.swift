//
//  SecondVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-17.
//

import UIKit

class SecondVC: UIViewController {
    var btnFirst:UIButton!
    
    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        btnFirst = UIButton( type: .system,
                              primaryAction: UIAction(title: "Go to First", handler: { _ in
                                AHXMain.shared.topVC( "FirstVC")
        }))
        let v = btnFirst!
        AHXLayout.width( v, 0.33 * AHC.w)
        AHXLayout.height( v, AHC.btnHeight)
        AHXLayout.center( v, AHC.w/2)
        AHXLayout.middle( v, AHC.h/2)
        self.view.addSubview( v)
    } // viewDidLoad()
    
} // class SecondVC
