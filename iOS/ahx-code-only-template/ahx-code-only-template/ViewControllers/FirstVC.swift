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
    
    //-----------------
    func layout() {
        let inset = AHXMain.scene.win.safeAreaInsets
        let w = UIScreen.main.bounds.width
        let h = UIScreen.main.bounds.height - inset.top - inset.bottom
        let bottom = UIScreen.main.bounds.height - inset.bottom
        let top = inset.top

        let v = btnSecond!
        AHXLayout.width( v, 0.33 * w)
        AHXLayout.height( v, AHXConstants.btnHeight)
        AHXLayout.center( v, w/2)
        AHXLayout.middle( v, h/2)
        self.view.addSubview( v)
    } // layout()

//    //------------------------------------------------
//    override func viewWillAppear(_ animated: Bool) {
//        super.viewWillAppear( animated)
//        AHXUtils.addSharedView( self, BottomNavVC.shared)
//    }
} // class FirstVC

