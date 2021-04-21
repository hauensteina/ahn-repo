//
//  MiddleVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-21.
//

// Content occupying most of my app screen

import UIKit

//==========================================
class MiddleVC: UIViewController {
    var lbInfo:UILabel!
    var nav:UINavigationController!
    var vcOne:FirstVC!
    var vcTwo:SecondVC!
    
    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        let rootVC = UIViewController() // This never gets popped off
        nav = UINavigationController( rootViewController: rootVC)
        self.addChild( nav)
        view.addSubview( nav.view)
        //nav.layout()
        nav.didMove( toParent: self)
        // Layout root frame
        let inset = AHXMain.scene.win.safeAreaInsets
        let w = UIScreen.main.bounds.width
        let h = UIScreen.main.bounds.height - inset.top - inset.bottom
        let bottom = UIScreen.main.bounds.height - inset.bottom
        let top = inset.top
        let v = self.view!
        AHXLayout.width( v, w)
        AHXLayout.height( v, h - 2 * 0.1 * h)
        AHXLayout.left( v, 0)
        AHXLayout.top( v, top + 0.1 * h)
    } // viewDidLoad()
} // class MiddleVC

