//
//  TopVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-21.
//

// Navigation bar at the top of the screen

import UIKit

//=================================
class TopVC: UIViewController {
    var btnBack:UIButton!
    
    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
    } // viewDidLoad()
    
    //-------------------------------------------------
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        layout()
    } // viewWillAppear()
    
    //--------------------------
    override func layout() {
        // Layout root frame
        var v = self.view!
        AHXLayout.width( v, AHC.w)
        AHXLayout.height( v, AHC.top_nav_height)
        AHXLayout.left( v, 0)
        AHXLayout.top( v, AHC.top)
        
        // Layout view components
        btnBack = UIButton( type: .system,
                            primaryAction: UIAction(title: "Back", handler: { _ in
                              AHXMain.shared.popVC()
                            }))
        v = btnBack
        self.view.addSubview( v)
        AHXLayout.width( v, v.intrinsicContentSize.width)
        AHXLayout.height( v, view.frame.height)
        AHXLayout.left( v, AHC.lmarg)
        AHXLayout.middle( v, view.frame.height / 2)
        //AHXLayout.border( v, .blue)
    } // layout()
} // class TopVC

