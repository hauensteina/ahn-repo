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
    
    //-----------------
    func layout() {
        let inset = AHXMain.scene.win.safeAreaInsets
        let w = UIScreen.main.bounds.width
        let h = UIScreen.main.bounds.height - inset.top - inset.bottom
        let bottom = UIScreen.main.bounds.height - inset.bottom
        let top = inset.top
        
        // Layout root frame
        var v = self.view!
        AHXLayout.width( v, w)
        AHXLayout.height( v, 0.1 * h)
        AHXLayout.left( v, 0)
        AHXLayout.top( v, top)
        
        // Layout view components
        btnBack = UIButton( type: .system,
                            primaryAction: UIAction(title: "Back", handler: { _ in
                              AHXMain.shared.popVC()
                            }))
        v = btnBack
        self.view.addSubview( v)
        AHXLayout.width( v, v.intrinsicContentSize.width)
        AHXLayout.height( v, view.frame.height)
        AHXLayout.left( v, 0.05 * w)
        AHXLayout.middle( v, view.frame.height / 2)
        //AHXLayout.border( v, .blue)
    } // layout()
} // class TopVC

