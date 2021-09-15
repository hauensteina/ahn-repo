//
//  AHXTableView.swift
//  ahx-tiles
//
//  Created by Andreas Hauenstein on 2021-06-07.
//

/*
 A less annoying way to define a tableview. Just pass it a bunch of views, and the space between
 them. The views should have buttons etc and react appropriately.
 
 Example usage:
 override func viewDidLoad() {
     super.viewDidLoad()
     self.tbv = AHXTableView( space:10.0)
     self.view.addSubview( tbv)
 }
 override func layout() {
     self.tbvWidth = self.view.frame.width
     self.tbvHeight = self.view.frame.height
     createViews()
     self.tbv.setViews( views: self.views)
     AHL.width( self.tbv, self.tbvWidth)
     AHL.height( self.tbv, self.tbvHeight)
 } // layout()
 
 */

import UIKit

//===========================================================================
class AHXTableView: UITableView, UITableViewDelegate, UITableViewDataSource {
    var views:[UIView]!
    var space:CGFloat!
    
    // We are a tableview who is its own delegate
    //===============================================
    init( space:CGFloat=10) {
        super.init( frame:CGRect(), style: .plain)
        self.delegate = self
        self.dataSource = self
        self.space = space
        // Magic to make dequeueReusableCell() work
        self.register( UITableViewCell.self, forCellReuseIdentifier: "AHXTableViewCell")
    } // init()
    
    // One view per tableview row
    //----------------------------------
    func setViews( views:[UIView]) {
        self.views = views
    } // setViews()
    
    // The number items in the tableview
    //------------------------------------------------------------------------------------
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return self.views.count
    } // numberOfRowsInSection()

    // Height of a row
    //-----------------------------------------------------------------------------------------
    func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
        let idx = indexPath.row
        let res = self.views[idx].frame.height + self.space
        return res
    } // heightForRow()

    //------------------------------------------------------------------------------------------------
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell
    {
        let idx = indexPath.row
        let aCell = tableView.dequeueReusableCell( withIdentifier: "AHXTableViewCell", for: indexPath)
        for s in aCell.contentView.subviews {
            s.removeFromSuperview()
        }
        AHL.subcenter( views[idx], aCell.contentView)
        AHL.submiddle( views[idx], aCell.contentView)
        aCell.selectionStyle = .none
        aCell.backgroundColor = .clear
        return aCell
    } // cellForRowAtIndexPath()
    
    // Define menu actions here
    //---------------------------------------------------------------------------------
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        // Do nothing. The subviews have actions.
    } // didSelectRowAtIndexPath()

    //------------------------------------------------------------
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
} // class AHXTableView
