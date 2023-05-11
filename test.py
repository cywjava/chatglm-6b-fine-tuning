# coding=UTF-8
import re


def find_response_items(content: str, extract: bool = True):
    """
    查找文本中的建议细项 比如 1. xxx 2. xxx 3. xxxx 1、xxxx 2 、xxxx 3、xxxx
    :param content: 源文本
    :param extract: 是否提取摘要、主旨 ，也就是每个细项的第一句话。
    :return:
    """
    content = content.replace("\\r\\n|\\n|\\r|\\s", "")
    matchNum = re.findall("\\d.", content)
    contentList = []
    for idx in range(len(matchNum)):
        if idx == len(matchNum) - 1:
            itemContent = content[content.index(matchNum[idx]):]
        else:
            itemContent = content[content.index(matchNum[idx]):content.index(matchNum[idx + 1])]
        if extract:
            itemContent = itemContent.replace(matchNum[idx], "").strip()
            symbols = re.findall("\\、|\\ 、|\\:|\\：|\\。|\\.|\\,|\\，|\\s|\\r\\n", itemContent)
            if len(symbols) > 0:
                itemContent = itemContent[0:itemContent.index(symbols[0])]

        contentList.append(itemContent)
    return contentList


str = "随着移动通信网络的不断扩大和普及，" \
      "运营商在移动通信网络运营过程中产生的数据量也急剧增加。" \
      "   这些数据对于运营商来说具有重要的商业   价值，可以帮助他们更好地了解用户行为、市场趋势和竞争对手，进而优化网络运营、提高服务质量和增加收入。在当前大数据技术尚未完全成熟的情况下，运营商大数据行业应用位置平台的建设可以为其提供一个数据存储、处理和分析的平台，从而帮助其更好地挖掘数据的价值。此外，该项目还可以为运营商提供数据可视化和报表生成等功能，以便他们更好地了解用户行为和市场趋势，从而制定更加精准的营销策略和决策。建议如下：1. 选择合适的技术平台：由于运营商大数据行业应用位置平台需要处理海量的数据，因此需要选择合适的技术平台。建议选择具有高可靠性、高性能和易扩展性的技术平台，例如云计算平台、大数据存储和处理引擎等。2. 建立完善的数据治理和隐私保护机制：由于运营商大数据行业应用位置平台需要处理大量的敏感数据，因此需要建立完善的数据治理和隐私保护机制，以确保数据的安全和隐私的保护。3. 加强数据可视化和报表生成功能：通过数据可视化和报表生成功能，运营商可以更好地了解用户行为和市场趋势，从而制定更加精准的营销策略和决策。建议选择具有可视化和报表生成功能的技术和工具，例如Tableau等。4. 建立开放的数据平台：开放的数据平台可以促进数据的共享和流通，从而为运营商带来更多的商业机会和收益。建议选择开放的数据平台，例如Facebook、Google等。5. 加强人才队伍建设：运营商大数据行业应用位置平台的建设需要专业的人才支持，因此需要加强人才队伍建设，包括招募专业的数据分析师、数据工程师等，以及提供良好的培训和支持。"
print(find_response_items(str))
