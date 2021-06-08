//
//  SecondVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-17.
//

import UIKit

class SecondVC: UIViewController {
    var images = [UIImage]()
    var views = [UIView]()
    var tbv:AHXTableView!
    var tbvWidth:CGFloat!
    var tbvHeight:CGFloat!

    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        loadImages()
        self.tbv = AHXTableView( space:10.0)
        self.view.addSubview( tbv)
    } // viewDidLoad()
    
    //-------------------------
    override func layout() {
        self.tbvWidth = self.view.frame.width
        self.tbvHeight = self.view.frame.height
        createViews()
        self.tbv.setViews( views: self.views)
        AHL.width( self.tbv, self.tbvWidth)
        AHL.height( self.tbv, self.tbvHeight)
    } // layout()
    
    // Load Images into our imges array
    //-------------------------------------
    func loadImages() {
        for idx in 1...5 {
            let iname = "photo\(idx)"
            let img = UIImage( named: iname)
            self.images.append( img!)
        } // for
    } // loadImages()
    
    //---------------------
    func createViews() {
        let viewHeight = AHC.h / 3.0
        let imgWidth = tbvWidth * 0.95
        let imgHeight = viewHeight * 0.95
        for (_,img) in images.enumerated() {
            let iv = UIImageView( image: img)
            AHL.height( iv, imgHeight)
            AHL.scaleWidth( iv, likeImage: img)
            if iv.frame.width > imgWidth {
                AHL.width( iv, imgWidth)
                AHL.scaleHeight( iv, likeImage: img)
            }
            // Container for the image
            let v = UIView()
            AHL.width( v, tbvWidth)
            AHL.height( v, iv.frame.height * 1.05)
            AHL.submiddle( iv, v)
            AHL.subcenter( iv, v)
            self.views.append( v)
        } // for
    } // createViews()
} // class SecondVC
