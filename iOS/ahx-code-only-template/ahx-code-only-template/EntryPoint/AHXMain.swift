//
//  AHXMain.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-17.
//

import UIKit

//=====================
class AHXMain {
    static let shared = AHXMain()
    static var app = UIApplication.shared.delegate as! AppDelegate
    static var scene = SceneDelegate.shared!
    var navVC:UINavigationController!
    var VCs = [String:UIViewController]()
    
    // Entry Point to App, called from SceneDelegate
    //------------------------------------------------
    func main( navVC:UINavigationController) {
        
        self.navVC = navVC
        navVC.setNavigationBarHidden( true, animated:false)
        VCs = [
            "FirstVC": FirstVC(),
            "SecondVC": SecondVC(),
            "BottomNavVC": BottomNavVC()
        ]
        // Force instantiation
        let _ = VCs["BottomNavVC"]!.view
        
        pushVC( "FirstVC")
        //AHXMain.scene = VCs["FirstVC"]!.view.window!.windowScene!.delegate
    } // main()
    
    // Push a VC by name
    //-------------------------------
    func pushVC( _ vcName:String) {
        let vc = VCs[vcName]!
        self.navVC.pushViewController( vc, animated: true)
    } // pushCV()

    // Replace top VC
    //-------------------------------
    func topVC( _ vcName:String) {
        let vc = VCs[vcName]!
        self.navVC.popViewController( animated:true)
        self.navVC.pushViewController( vc, animated: true)
    } // topVC()
    
} // class AHXMain
