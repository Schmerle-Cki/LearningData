##### 代码运行说明

【以下命令均在根目录下执行】

##### image fusion

1. `python dot_mask.py``
2. ``python fusion.py`

生成图片在 `image fusion`文件夹



##### face morphing

【原图片、中间素材、生成图均在 `face morphing`目录】

1. `hand_label.py`中已注释掉对除source2外所有图片的手工标注（默认情况下不允许修改已经标注的图片）；运行该文件生成 `image_name + hand_points.json`
2. [建议用已经提取好的文件]`python PointsExtractor.py `,生成 `image_name+array.json`(特征点集合)
3.  `python morphing.py`: 生成中间文件（很多：各参数下的warp图、三角片图）以及最终结果（文件名含final）



##### view morphing

1. 前两步同 `face morphing`（若运行，请删掉`__main__`中各行前的 `#`）
2.  `python view_morphing.py`（会生成`Hs`变换前的图像，请手工标注图片的四个顶点（顺序：左上，左下，右上，右下）



【注：狮子、`view morphing target2`标注前68个点顺序参考如下】

左脸上方（8）——顺着脸颊（1）——右脸上方（8）——左眉毛（5）——右眉毛（5）—— 鼻子（上到下）（4）——鼻翼（左到右）（5）——左眼（左到右到左）（6）——右眼（6）——嘴唇外圈（左到右到左，上7下5）——嘴唇内圈（左到右5个，右到左3个）

